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

/// Handlers for events from the wprs server.
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::os::fd::OwnedFd;
use std::thread;

use smithay_client_toolkit::data_device_manager::data_offer;
use smithay_client_toolkit::reexports::client::Proxy;
use smithay_client_toolkit::reexports::client::protocol::wl_data_device_manager::DndAction;
use smithay_client_toolkit::shell::WaylandSurface;

use crate::client::RemoteSurface;
use crate::client::RemoteXdgPopup;
use crate::client::RemoteXdgToplevel;
use crate::client::Role;
use crate::client::WprsClientState;
use crate::client::subsurface;
use crate::client::subsurface::RemoteSubSurface;
use crate::fallible_entry::FallibleEntryExt;
use crate::filtering;
use crate::filtering::ValidatedTileStream;
use crate::prelude::*;
use crate::serialization::Capabilities;
use crate::serialization::ClientId;
use crate::serialization::Event;
use crate::serialization::RecvType;
use crate::serialization::Request;
use crate::serialization::SendType;
use crate::serialization::framing::CompressedTileStream;
use crate::serialization::wayland;
use crate::serialization::wayland::BufferAssignment;
use crate::serialization::wayland::ClientSurface;
use crate::serialization::wayland::CursorImage;
use crate::serialization::wayland::CursorImageStatus;
use crate::serialization::wayland::DataDestinationRequest;
use crate::serialization::wayland::DataEvent;
use crate::serialization::wayland::DataRequest;
use crate::serialization::wayland::DataSource;
use crate::serialization::wayland::DataSourceEvent;
use crate::serialization::wayland::DataSourceRequest;
use crate::serialization::wayland::DataToTransfer;
use crate::serialization::wayland::DataTransferTarget;
use crate::serialization::wayland::SurfaceEvent;
use crate::serialization::wayland::SurfaceEventPayload;
use crate::serialization::wayland::SurfaceRequest;
use crate::serialization::wayland::SurfaceRequestPayload;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::WlSurfaceId;
use crate::serialization::xdg_shell;
use crate::serialization::xdg_shell::PopupRequest;
use crate::serialization::xdg_shell::PopupRequestPayload;
use crate::serialization::xdg_shell::ToplevelRequest;
use crate::serialization::xdg_shell::ToplevelRequestPayload;

impl WprsClientState {
    fn request_buffer_resync(&mut self, client_id: ClientId, surface_id: WlSurfaceId) {
        if let Some(client) = self.remote_display.clients.get_mut(&client_id)
            && let Some(remote_surface) = client.surfaces.get_mut(&surface_id)
        {
            remote_surface.drop_buffer();
        }
        self.serializer
            .writer()
            .send(SendType::Object(Event::Surface(SurfaceEvent {
                surface_id,
                payload: SurfaceEventPayload::BufferResyncRequested,
            })));
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_commit(
        &mut self,
        client_id: ClientId,
        surface_id: WlSurfaceId,
        mut surface_state: SurfaceState,
    ) -> Result<()> {
        let had_new_buffer = surface_state
            .buffer
            .as_ref()
            .is_some_and(BufferAssignment::is_new);
        let buffer_removed = matches!(surface_state.buffer, Some(BufferAssignment::Removed));

        let frame_callback_completed = {
            let client = self.remote_display.client(&client_id);
            let remote_surface = match client
                .surfaces
                .entry(surface_id)
                .or_insert_with_result(|| {
                    RemoteSurface::new(
                        client.id,
                        surface_id,
                        &self.compositor_state,
                        &self.qh,
                        &mut self.object_bimap,
                    )
                })
                .and_then(|remote_surface| {
                    remote_surface
                        .apply_buffer(surface_state.buffer.take())
                        .location(loc!())?;
                    Ok(remote_surface)
                }) {
                Ok(remote_surface) => remote_surface,
                Err(err) => {
                    self.request_buffer_resync(client_id, surface_id);
                    return Err(err).location(loc!());
                },
            };

            remote_surface.set_transformation(
                surface_state.buffer_scale,
                surface_state.buffer_transform.map(Into::into),
            );

            remote_surface.set_viewport_state(
                surface_state.viewport_state,
                &self.wp_viewporter,
                &self.qh,
            );

            remote_surface
                .set_input_region(surface_state.input_region.take(), &self.compositor_state)
                .location(loc!())?;
            remote_surface
                .set_opaque_region(surface_state.opaque_region.take(), &self.compositor_state)
                .location(loc!())?;

            remote_surface
                .frame_damage
                .merge_commit(surface_state.damage.take(), had_new_buffer);
            if let Some(scroll_damage) = remote_surface.pending_scroll_damage.take() {
                remote_surface.frame_damage.ensure_covered(scroll_damage);
            }
            remote_surface.add_buffer_delta(surface_state.buffer_delta.take());

            remote_surface.frame_callback_completed
        };

        let primary_seat = matches!(surface_state.role, Some(wayland::Role::XdgPopup(_)))
            .then(|| self.primary_seat())
            .flatten();
        let client = self.remote_display.client(&client_id);
        let surfaces = &mut client.surfaces;

        subsurface::populate_subsurfaces(
            client.id,
            surface_id,
            &surface_state.z_ordered_children,
            surfaces,
            &self.compositor_state,
            &self.subcompositor,
            &self.qh,
            &mut self.object_bimap,
        )
        .location(loc!())?;
        subsurface::reorder_subsurfaces(surface_id, &surface_state, surfaces).location(loc!())?;

        match &surface_state.role {
            Some(wayland::Role::Cursor(_)) => {},
            Some(wayland::Role::SubSurface(_)) => RemoteSubSurface::apply(
                client.id,
                surface_state,
                surface_id,
                surfaces,
                &self.compositor_state,
                &self.subcompositor,
                &self.qh,
                &mut self.object_bimap,
            )
            .location(loc!())?,
            Some(wayland::Role::XdgToplevel(_)) => RemoteXdgToplevel::apply(
                surface_state,
                surface_id,
                surfaces,
                &self.xdg_shell_state,
                &self.output_state,
                &self.qh,
                &self.title_prefix,
            )
            .location(loc!())?,
            Some(wayland::Role::XdgPopup(_)) => RemoteXdgPopup::apply(
                surface_state,
                surface_id,
                surfaces,
                &self.xdg_shell_state,
                primary_seat.as_ref(),
                &self.qh,
            )
            .location(loc!())?,
            None => {
                subsurface::detach_subsurface(surface_id, surfaces);
            },
        }

        if buffer_removed && let Some(remote_surface) = surfaces.get_mut(&surface_id) {
            match &mut remote_surface.role {
                Some(Role::XdgToplevel(toplevel)) => toplevel.configured = false,
                Some(Role::XdgPopup(popup)) => popup.configured = false,
                _ => {},
            }
        }

        if frame_callback_completed {
            subsurface::commit_sync_children(surface_id, surfaces).location(loc!())?;
            let remote_surface = surfaces.get_mut(&surface_id).location(loc!())?;
            match &remote_surface.role {
                Some(Role::SubSurface(subsurface)) if subsurface.sync => {},
                Some(Role::XdgToplevel(toplevel)) if !toplevel.configured => {
                    toplevel.commit();
                },
                Some(Role::XdgPopup(popup)) if !popup.configured => {
                    popup.commit();
                },
                _ => remote_surface
                    .draw_buffer_send_frame(&self.qh)
                    .location(loc!())?,
            }
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_surface_destroy(&mut self, client_id: ClientId, surface_id: WlSurfaceId) {
        self.current_cursor_image.take_if(|img| {
            matches!(
                &img.status,
                CursorImageStatus::Surface { client_surface, .. }
                    if client_surface.client == client_id
                        && client_surface.surface == surface_id
            )
        });
        self.object_bimap.remove_by_left(&(client_id, surface_id));
        if let Some(client) = self.remote_display.clients.get_mut(&client_id) {
            subsurface::destroy_surface(surface_id, &mut client.surfaces);
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_surface(&mut self, request: SurfaceRequest) -> Result<()> {
        match request.payload {
            SurfaceRequestPayload::Commit(surface_state) => {
                self.handle_commit(request.client, request.surface, surface_state)
                    .location(loc!())?;
            },
            SurfaceRequestPayload::Destroyed => {
                self.handle_surface_destroy(request.client, request.surface);
            },
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_toplevel(&mut self, request: ToplevelRequest) -> Result<()> {
        if matches!(request.payload, ToplevelRequestPayload::Destroyed) {
            // TODO(https://github.com/wayland-transpositor/wprs/pull/114):
            // remove this workaround once sctk allows destroying `Window`
            // without destroying its underlying `WlSurface`. Do not drop
            // `surface.role` before `handle_surface_destroy`: child
            // `wl_subsurface`s must be destroyed before the parent `wl_surface`.
            self.handle_surface_destroy(request.client, request.surface);
            return Ok(());
        }

        let primary_seat = matches!(
            request.payload,
            ToplevelRequestPayload::ShowWindowMenu { .. }
        )
        .then(|| self.primary_seat())
        .flatten();

        let Some(client) = self.remote_display.clients.get_mut(&request.client) else {
            return Ok(());
        };
        // TODO: these properties aren't in the double-buffered state in
        // smithay, but still only take affect on commit. That seems wrong. In
        // the meantime though, we can get these before the initial commit.
        let Ok(surface) = client.surface(&request.surface) else {
            warn!("received request for unknown surface");
            return Ok(());
        };

        if let Some(Role::XdgToplevel(toplevel)) = &surface.role {
            match request.payload {
                ToplevelRequestPayload::Destroyed => unreachable!(),
                ToplevelRequestPayload::SetMaximized => {
                    toplevel.local_window.set_maximized();
                },
                ToplevelRequestPayload::UnsetMaximized => {
                    toplevel.local_window.unset_maximized();
                },
                ToplevelRequestPayload::SetFullscreen(output_id) => {
                    let output =
                        output_id.and_then(|id| crate::client::find_output(&self.output_state, id));
                    toplevel.local_window.set_fullscreen(output.as_ref());
                },
                ToplevelRequestPayload::UnsetFullscreen => {
                    toplevel.local_window.unset_fullscreen();
                },
                ToplevelRequestPayload::SetMinimized => {
                    toplevel.local_window.set_minimized();
                },
                ToplevelRequestPayload::Move(xdg_shell::Move { serial }) => {
                    toplevel
                        .local_window
                        .xdg_toplevel()
                        ._move(&self.seat_state.seats().next().location(loc!())?, serial);
                },
                ToplevelRequestPayload::Resize(xdg_shell::Resize { serial, edge }) => {
                    toplevel.local_window.xdg_toplevel().resize(
                        &self.seat_state.seats().next().location(loc!())?,
                        serial,
                        // The error type is (). :(
                        edge.try_into()
                            .map_err(|_| anyhow!("invalid edge"))
                            .location(loc!())?,
                    );
                },
                ToplevelRequestPayload::ShowWindowMenu { serial, location } => {
                    let seat = primary_seat.location(loc!())?;
                    toplevel
                        .local_window
                        .show_window_menu(&seat, serial, (location.x, location.y));
                },
            }
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_popup(&mut self, request: PopupRequest) -> Result<()> {
        match request.payload {
            PopupRequestPayload::Destroyed => {
                // TODO(https://github.com/wayland-transpositor/wprs/pull/114):
                // remove this workaround once sctk allows destroying `Popup`
                // without destroying its underlying `WlSurface`. Do not drop
                // `surface.role` before `handle_surface_destroy`: child
                // `wl_subsurface`s must be destroyed before the parent `wl_surface`.
                self.handle_surface_destroy(request.client, request.surface);
            },
        }
        Ok(())
    }

    fn ensure_cursor_surface(&mut self, client_surface: ClientSurface) -> Result<()> {
        let client = self.remote_display.client(&client_surface.client);
        let remote_surface = client
            .surfaces
            .entry(client_surface.surface)
            .or_insert_with_result(|| {
                RemoteSurface::new(
                    client.id,
                    client_surface.surface,
                    &self.compositor_state,
                    &self.qh,
                    &mut self.object_bimap,
                )
            })
            .location(loc!())?;
        remote_surface.role = Some(Role::Cursor);
        Ok(())
    }

    pub(crate) fn apply_current_cursor_image(&mut self) -> Result<()> {
        let Some(cursor_image) = self.current_cursor_image.clone() else {
            return Ok(());
        };
        let Some(seat_obj) = self.seat_objects.last() else {
            return Ok(());
        };
        let Some(themed_pointer) = seat_obj.pointer.as_ref() else {
            warn!("State has no pointer capability, ignoring cursor image.");
            return Ok(());
        };

        match cursor_image.status {
            CursorImageStatus::Named(name) => {
                themed_pointer
                    .set_cursor(
                        &self.conn,
                        name.parse()
                            .with_context(loc!(), || format!("Unknown cursor name {name:?}."))?,
                    )
                    .location(loc!())?;
            },
            CursorImageStatus::Surface {
                client_surface,
                hotspot,
            } => {
                let cursor_wl_surface = self
                    .remote_display
                    .clients
                    .get(&client_surface.client)
                    .and_then(|client| client.surfaces.get(&client_surface.surface))
                    .map(|surface| surface.wl_surface().clone())
                    .context(loc!(), "current_cursor_image references unknown surface")?;
                themed_pointer.pointer().set_cursor(
                    self.last_enter_serial,
                    Some(&cursor_wl_surface),
                    hotspot.x,
                    hotspot.y,
                );
            },
            CursorImageStatus::Hidden => {
                themed_pointer.hide_cursor().location(loc!())?;
            },
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_cursor_image(&mut self, cursor_image: CursorImage) -> Result<()> {
        if let CursorImageStatus::Surface { client_surface, .. } = cursor_image.status {
            self.ensure_cursor_surface(client_surface)
                .location(loc!())?;
        }
        self.current_cursor_image = Some(cursor_image);
        self.apply_current_cursor_image()
    }

    fn send_dnd_cancelled(&self, source_id: u64) {
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                DataSourceEvent::DnDCancelled { source_id },
            ))));
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_data(&mut self, data: DataRequest) -> Result<()> {
        match data {
            DataRequest::SourceRequest(DataSourceRequest::StartDrag(start_drag)) => {
                if self.pressed_buttons.is_empty() {
                    debug!(
                        "cancelling StartDrag {} because no pointer button is pressed on host",
                        start_drag.source_id
                    );
                    self.send_dnd_cancelled(start_drag.source_id);
                    return Ok(());
                }

                let Some(origin_surface) = self
                    .remote_display
                    .clients
                    .get(&start_drag.origin.client)
                    .and_then(|client| client.surfaces.get(&start_drag.origin.surface))
                    .map(|surface| surface.wl_surface().clone())
                else {
                    warn!(
                        "cancelling StartDrag {} for unknown origin surface {:?}",
                        start_drag.source_id, start_drag.origin
                    );
                    self.send_dnd_cancelled(start_drag.source_id);
                    return Ok(());
                };

                let Ok(dnd_actions) = DndAction::try_from(start_drag.source.dnd_actions) else {
                    warn!(
                        "cancelling StartDrag {} with invalid dnd actions {}",
                        start_drag.source_id, start_drag.source.dnd_actions
                    );
                    self.send_dnd_cancelled(start_drag.source_id);
                    return Ok(());
                };

                // TODO: support multiple seats
                if self.seat_objects.is_empty() {
                    self.send_dnd_cancelled(start_drag.source_id);
                    return Ok(());
                }

                let icon_surface = match start_drag.icon {
                    None => None,
                    Some(ClientSurface { client, surface }) => {
                        let client = self.remote_display.client(&client);
                        let remote_surface =
                            match client.surfaces.entry(surface).or_insert_with_result(|| {
                                RemoteSurface::new(
                                    client.id,
                                    surface,
                                    &self.compositor_state,
                                    &self.qh,
                                    &mut self.object_bimap,
                                )
                            }) {
                                Ok(remote_surface) => remote_surface,
                                Err(err) => {
                                    self.send_dnd_cancelled(start_drag.source_id);
                                    return Err(err).location(loc!());
                                },
                            };
                        Some(remote_surface.wl_surface().clone())
                    },
                };

                let seat_obj = self.seat_objects.last().unwrap();

                let source = self.data_device_manager_state.create_drag_and_drop_source(
                    &self.qh,
                    start_drag.source.mime_types.iter().map(String::as_str),
                    dnd_actions,
                );
                source.start_drag(
                    &seat_obj.data_device,
                    &origin_surface,
                    icon_surface.as_ref(),
                    start_drag.serial,
                );
                self.dnd_source = Some((start_drag.source_id, source));
                self.dnd_source_dropped = false;
            },
            DataRequest::SourceRequest(DataSourceRequest::CancelDrag(source_id)) => {
                if self
                    .dnd_source
                    .as_ref()
                    .is_some_and(|(id, _)| *id == source_id)
                {
                    // Dropping `DragSource` destroys the host `WlDataSource`.
                    self.dnd_source = None;
                    self.dnd_source_dropped = false;
                    self.pressed_buttons.clear();
                }
            },
            DataRequest::SourceRequest(DataSourceRequest::SetSelection(
                source,
                mut source_metadata,
            )) => {
                match source {
                    DataSource::Selection => {
                        // TODO: support multiple seats
                        if let (Some(seat_obj), Some(serial)) = (
                            self.seat_objects.last(),
                            self.last_implicit_grab_serial.take(),
                        ) {
                            source_metadata.mime_types.push("_wprs_marker".to_string());
                            let mime_types = source_metadata.mime_types.iter().map(String::as_str);
                            let source = self
                                .data_device_manager_state
                                .create_copy_paste_source(&self.qh, mime_types);
                            source.set_selection(&seat_obj.data_device, serial);
                            self.selection_source = Some(source);
                        }
                    },
                    DataSource::Primary => {
                        if let (Some(seat_obj), Some(serial)) =
                            (self.seat_objects.last(), self.last_mouse_down_serial.take())
                            && let (
                                Some(primary_selection_manager_state),
                                Some(primary_selection_device),
                            ) = (
                                &self.primary_selection_manager_state,
                                &seat_obj.primary_selection_device,
                            )
                        {
                            source_metadata.mime_types.push("_wprs_marker".to_string());
                            let mime_types = source_metadata.mime_types.iter().map(String::as_str);
                            let source = primary_selection_manager_state
                                .create_selection_source(&self.qh, mime_types);
                            source.set_selection(primary_selection_device, serial);
                            self.primary_selection_source = Some(source);
                        }
                    },
                }
            },
            DataRequest::DestinationRequest(DataDestinationRequest::DnDAcceptMimeType {
                offer_id,
                mime_type,
            }) => {
                if let Some(dnd_offer) = self.dnd_offer_by_id(offer_id) {
                    // Use the host enter serial and call the raw `wl_data_offer`
                    // directly so post-drop `DndAction::Ask` accepts also go
                    // through after `wl_data_device.leave`.
                    dnd_offer.inner().accept(dnd_offer.serial, mime_type);
                }
            },
            DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer {
                target,
                mime_type,
                transfer_id,
            }) => {
                let read_pipe = match target {
                    DataTransferTarget::Primary => self
                        .primary_selection_offer
                        .as_ref()
                        .and_then(|offer| offer.receive(mime_type.clone()).ok()),
                    DataTransferTarget::Selection => self
                        .selection_offer
                        .as_ref()
                        .and_then(|offer| offer.receive(mime_type.clone()).ok()),
                    DataTransferTarget::DnD(offer_id) => {
                        self.dnd_offer_by_id(offer_id).and_then(|offer| {
                            data_offer::receive(offer.inner(), mime_type.clone()).ok()
                        })
                    },
                };
                let writer = self.serializer.writer();
                if let Some(mut read_pipe) = read_pipe {
                    debug!("spawning receive thread for mime {mime_type} transfer {transfer_id}");
                    // The data source application will write to the other end
                    // of read_pipe at its convenience and then close the file
                    // descriptor, so spawn off a thread to perform that read
                    // and send the data to the server whenever the read is
                    // completed. Always reply (even with empty bytes on error)
                    // so `wprsd` closes the requesting client's fd.
                    thread::spawn(move || {
                        debug!("in receive thread for mime {mime_type} transfer {transfer_id}");
                        let mut buf = Vec::new();
                        let bytes_read = read_pipe.read_to_end(&mut buf);
                        debug!("read selection ({bytes_read:?} bytes): {buf:?}");
                        writer.send(SendType::Object(Event::Data(DataEvent::TransferData(
                            transfer_id,
                            DataToTransfer(buf),
                        ))));
                    });
                } else {
                    writer.send(SendType::Object(Event::Data(DataEvent::TransferData(
                        transfer_id,
                        DataToTransfer(Vec::new()),
                    ))));
                }
            },
            DataRequest::DestinationRequest(DataDestinationRequest::DnDSetDestinationActions {
                offer_id,
                dnd_actions,
                preferred_action,
            }) => {
                if let Some(dnd_offer) = self.dnd_offer_by_id(offer_id) {
                    let dnd_actions: DndAction = dnd_actions
                        .try_into()
                        .map_err(|()| anyhow!("invalid dnd actions"))
                        .location(loc!())?;
                    let preferred_action: DndAction = preferred_action
                        .try_into()
                        .map_err(|()| anyhow!("invalid preferred dnd action"))
                        .location(loc!())?;
                    if dnd_offer.inner().version() >= 3 {
                        dnd_offer.inner().set_actions(dnd_actions, preferred_action);
                    }
                }
            },
            DataRequest::DestinationRequest(DataDestinationRequest::DnDFinish { offer_id }) => {
                if let Some(dnd_offer) = self.take_dnd_offer_by_id(offer_id) {
                    dnd_offer.finish();
                    dnd_offer.destroy();
                }
            },
            DataRequest::DestinationRequest(DataDestinationRequest::DnDCancel { offer_id }) => {
                if let Some(dnd_offer) = self.take_dnd_offer_by_id(offer_id) {
                    dnd_offer.destroy();
                }
            },
            DataRequest::TransferData(transfer_id, data) => {
                let write_pipe = self
                    .pending_send_pipes
                    .remove(&transfer_id)
                    .location(loc!())?;
                if data.0.is_empty() {
                    return Ok(());
                }
                let fd = OwnedFd::from(write_pipe);
                let mut f = File::from(fd);
                // If data is large, the write may block if the reader (the
                // application requesting the data) isn't reading it quickly
                // enough, so do the write in a separate thread to avoid
                // blocking the event loop. The thread will then terminate.
                thread::spawn(move || {
                    f.write_all(&data.0).log_and_ignore(loc!());
                });
            },
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_client_disconnected(&mut self, client: ClientId) -> Result<()> {
        self.current_cursor_image.take_if(|img| {
            matches!(
                &img.status,
                CursorImageStatus::Surface { client_surface, .. }
                    if client_surface.client == client
            )
        });
        if let Some(mut remote_client) = self.remote_display.clients.remove(&client) {
            subsurface::detach_all_subsurfaces(&mut remote_client.surfaces);
        }
        self.object_bimap.retain(|(cid, _), _| *cid != client);
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_capabilities(&mut self, caps: Capabilities) -> Result<()> {
        self.tile_cache.clear();
        self.capabilities.set(caps).location(loc!())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_buffer(&mut self, buffer_stream: CompressedTileStream) -> Result<()> {
        let client_id = buffer_stream.header.client;
        let surface_id = buffer_stream.header.surface;
        let frame_seq = buffer_stream.header.frame_seq;
        if buffer_stream.header.reset_tile_cache {
            self.tile_cache.clear();
        }

        let decode_res = (|| -> Result<()> {
            let validated = ValidatedTileStream::new(buffer_stream).location(loc!())?;
            let metadata = validated.metadata();

            let client = self.remote_display.client(&client_id);
            let remote_surface = client
                .surfaces
                .entry(surface_id)
                .or_insert_with_result(|| {
                    RemoteSurface::new(
                        client.id,
                        surface_id,
                        &self.compositor_state,
                        &self.qh,
                        &mut self.object_bimap,
                    )
                })
                .location(loc!())?;

            let canvas = match remote_surface.prepare_buffer_canvas(
                metadata,
                validated.incremental_tile_rects(),
                validated.scroll_rect(),
                &mut self.pool,
                &mut self.slot_copy_scratch,
                &mut self.decompressor,
            ) {
                Ok(canvas) => canvas,
                Err(err) => {
                    // Apply connection-wide tile cache updates (`reset`, `evict`,
                    // and `CachePut` tiles) even when this surface cannot apply
                    // the frame so other surfaces and already-pipelined frames
                    // stay in exact lockstep with `wprsd`.
                    filtering::decompress_cache_puts_only(
                        validated,
                        &mut self.decompressor,
                        &mut self.tile_cache,
                    )
                    .log_and_ignore(loc!());
                    return Err(err).location(loc!());
                },
            };
            filtering::decompress_and_unfilter_streaming(
                validated,
                &mut self.decompressor,
                canvas,
                Some(&mut self.tile_cache),
            )
            .location(loc!())
        })();

        if let Err(err) = decode_res {
            // Request resync before acknowledging the credit so any deferred
            // flush triggered by `FrameAck` sends a full frame instead of an
            // incremental delta against corrupt state.
            self.request_buffer_resync(client_id, surface_id);
            self.serializer
                .writer()
                .send(SendType::Object(Event::FrameAck { seq: frame_seq }));
            return Err(err);
        }
        self.serializer
            .writer()
            .send(SendType::Object(Event::FrameAck { seq: frame_seq }));
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    pub fn handle_request(&mut self, request: RecvType<Request>) {
        self.record_server_activity();
        match request {
            RecvType::Connected => unreachable!("client never receives RecvType::Connected"),
            RecvType::Disconnected => unreachable!("client never receives RecvType::Disconnected"),
            RecvType::Object(Request::Surface(surface)) => self.handle_surface(surface),
            RecvType::Object(Request::Toplevel(toplevel)) => self.handle_toplevel(toplevel),
            RecvType::Object(Request::Popup(popup)) => self.handle_popup(popup),
            RecvType::Object(Request::CursorImage(cursor_image)) => {
                self.handle_cursor_image(cursor_image)
            },
            RecvType::Object(Request::Data(data)) => self.handle_data(data),
            RecvType::Object(Request::ClientDisconnected(client)) => {
                self.handle_client_disconnected(client)
            },
            RecvType::Object(Request::Capabilities(caps)) => self.handle_capabilities(caps),
            RecvType::Object(Request::HeartbeatAck { seq: _ }) => Ok(()),
            RecvType::RawBuffer(buffer) => self.handle_buffer(buffer),
        }
        .log_and_ignore(loc!())
        // TODO: maybe send errors back to the server.
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Read;
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use crossbeam_channel::Receiver;
    use nix::fcntl::OFlag;
    use nix::unistd;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_popup as server_xdg_popup;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_popup::XdgPopup as ServerXdgPopup;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_positioner::XdgPositioner as ServerXdgPositioner;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_surface as server_xdg_surface;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_surface::XdgSurface as ServerXdgSurface;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel as server_xdg_toplevel;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::XdgToplevel as ServerXdgToplevel;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_wm_base;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_wm_base::XdgWmBase as ServerXdgWmBase;
    use smithay::reexports::wayland_server::Client as ServerClient;
    use smithay::reexports::wayland_server::DataInit;
    use smithay::reexports::wayland_server::Dispatch as ServerDispatch;
    use smithay::reexports::wayland_server::Display;
    use smithay::reexports::wayland_server::DisplayHandle;
    use smithay::reexports::wayland_server::GlobalDispatch;
    use smithay::reexports::wayland_server::New;
    use smithay::reexports::wayland_server::Resource;
    use smithay::reexports::wayland_server::WEnum as ServerWEnum;
    use smithay::reexports::wayland_server::backend::ClientData;
    use smithay::reexports::wayland_server::backend::ClientId as ServerClientId;
    use smithay::reexports::wayland_server::backend::DisconnectReason;
    use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer as ServerWlBuffer;
    use smithay::reexports::wayland_server::protocol::wl_callback::WlCallback as ServerWlCallback;
    use smithay::reexports::wayland_server::protocol::wl_compositor;
    use smithay::reexports::wayland_server::protocol::wl_compositor::WlCompositor as ServerWlCompositor;
    use smithay::reexports::wayland_server::protocol::wl_data_device as server_data_device;
    use smithay::reexports::wayland_server::protocol::wl_data_device::WlDataDevice as ServerWlDataDevice;
    use smithay::reexports::wayland_server::protocol::wl_data_device_manager as server_ddm;
    use smithay::reexports::wayland_server::protocol::wl_data_device_manager::DndAction as ServerDndAction;
    use smithay::reexports::wayland_server::protocol::wl_data_device_manager::WlDataDeviceManager as ServerWlDataDeviceManager;
    use smithay::reexports::wayland_server::protocol::wl_data_offer as server_data_offer;
    use smithay::reexports::wayland_server::protocol::wl_data_offer::WlDataOffer as ServerWlDataOffer;
    use smithay::reexports::wayland_server::protocol::wl_data_source as server_data_source;
    use smithay::reexports::wayland_server::protocol::wl_data_source::WlDataSource as ServerWlDataSource;
    use smithay::reexports::wayland_server::protocol::wl_keyboard as server_keyboard;
    use smithay::reexports::wayland_server::protocol::wl_keyboard::WlKeyboard as ServerWlKeyboard;
    use smithay::reexports::wayland_server::protocol::wl_output as server_output;
    use smithay::reexports::wayland_server::protocol::wl_output::WlOutput as ServerWlOutput;
    use smithay::reexports::wayland_server::protocol::wl_pointer as server_pointer;
    use smithay::reexports::wayland_server::protocol::wl_pointer::WlPointer as ServerWlPointer;
    use smithay::reexports::wayland_server::protocol::wl_region::WlRegion as ServerWlRegion;
    use smithay::reexports::wayland_server::protocol::wl_seat as server_seat;
    use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat as ServerWlSeat;
    use smithay::reexports::wayland_server::protocol::wl_shm;
    use smithay::reexports::wayland_server::protocol::wl_shm::WlShm as ServerWlShm;
    use smithay::reexports::wayland_server::protocol::wl_shm_pool;
    use smithay::reexports::wayland_server::protocol::wl_shm_pool::WlShmPool as ServerWlShmPool;
    use smithay::reexports::wayland_server::protocol::wl_subcompositor;
    use smithay::reexports::wayland_server::protocol::wl_subcompositor::WlSubcompositor as ServerWlSubcompositor;
    use smithay::reexports::wayland_server::protocol::wl_subsurface as server_subsurface;
    use smithay::reexports::wayland_server::protocol::wl_subsurface::WlSubsurface as ServerWlSubsurface;
    use smithay::reexports::wayland_server::protocol::wl_surface as server_surface;
    use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface as ServerWlSurface;
    use smithay_client_toolkit::reexports::client::Connection;
    use smithay_client_toolkit::reexports::client::EventQueue;
    use smithay_client_toolkit::reexports::client::Proxy;
    use smithay_client_toolkit::reexports::client::globals::registry_queue_init;

    use super::*;
    use crate::client::ClientOptions;
    use crate::serialization::CompressionMode;
    use crate::serialization::Serializer;
    use crate::serialization::wayland::DataDestinationEvent;
    use crate::serialization::wayland::DataDestinationRequest;
    use crate::serialization::wayland::DataEvent;
    use crate::serialization::wayland::DataRequest;
    use crate::serialization::wayland::DataSourceEvent;
    use crate::serialization::wayland::DataSourceRequest;
    use crate::serialization::wayland::DataTransferTarget;
    use crate::serialization::wayland::SourceMetadata;
    use crate::serialization::wayland::StartDrag;

    struct DummyClientData;
    impl ClientData for DummyClientData {
        fn initialized(&self, _client_id: ServerClientId) {}
        fn disconnected(&self, _client_id: ServerClientId, _reason: DisconnectReason) {}
    }

    #[derive(Debug, PartialEq)]
    enum HostRecordedReq {
        StartDrag {
            origin_id: u32,
            has_icon: bool,
            serial: u32,
        },
        SourceOffer(String),
        SourceSetActions(ServerDndAction),
        SourceDestroy,
        OfferAccept {
            serial: u32,
            mime_type: Option<String>,
        },
        OfferSetActions {
            dnd_actions: ServerDndAction,
            preferred_action: ServerDndAction,
        },
        OfferReceive(String),
        OfferFinish,
        OfferDestroy,
    }

    #[derive(Debug, Eq, PartialEq)]
    enum HostXdgReq {
        SurfaceAttach {
            surface_id: u32,
            has_buffer: bool,
            x: i32,
            y: i32,
        },
        SurfaceOffset {
            surface_id: u32,
            x: i32,
            y: i32,
        },
        SurfaceCommit {
            surface_id: u32,
        },
        PopupGrab {
            surface_id: u32,
            serial: u32,
        },
        SubsurfacePlaceAbove {
            surface_id: u32,
            sibling_id: u32,
        },
        SubsurfacePlaceBelow {
            surface_id: u32,
            sibling_id: u32,
        },
        ToplevelShowWindowMenu {
            surface_id: u32,
            serial: u32,
            x: i32,
            y: i32,
        },
        ToplevelSetFullscreen {
            surface_id: u32,
            output_tag: Option<u32>,
        },
        ToplevelUnsetFullscreen {
            surface_id: u32,
        },
    }

    #[derive(Default)]
    struct MockHostState {
        client: Option<ServerClient>,
        keyboard: Option<ServerWlKeyboard>,
        pointer: Option<ServerWlPointer>,
        data_device: Option<ServerWlDataDevice>,
        surfaces: Vec<ServerWlSurface>,
        pending_buffers: HashMap<u32, ServerWlBuffer>,
        hold_committed_buffers: bool,
        held_buffers: Vec<ServerWlBuffer>,
        damage_buffers: Vec<(u32, i32, i32, i32, i32)>,
        pending_frame_callbacks: Vec<ServerWlCallback>,
        subsurface_stacks: HashMap<u32, Vec<u32>>,
        subsurface_positions: HashMap<u32, (i32, i32)>,
        xdg_popups: HashMap<u32, ServerXdgPopup>,
        last_source: Option<ServerWlDataSource>,
        receive_fds: Vec<(String, OwnedFd)>,
        reqs: Vec<HostRecordedReq>,
        xdg_reqs: Vec<HostXdgReq>,
    }

    impl GlobalDispatch<ServerWlShm, ()> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerWlShm>,
            _global_data: &(),
            data_init: &mut DataInit<'_, Self>,
        ) {
            let shm = data_init.init(resource, ());
            shm.format(wl_shm::Format::Argb8888);
            shm.format(wl_shm::Format::Xrgb8888);
        }
    }

    impl ServerDispatch<ServerWlShm, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlShm,
            request: wl_shm::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            if let wl_shm::Request::CreatePool { id, .. } = request {
                data_init.init(id, ());
            }
        }
    }

    impl ServerDispatch<ServerWlShmPool, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlShmPool,
            request: wl_shm_pool::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            if let wl_shm_pool::Request::CreateBuffer { id, .. } = request {
                data_init.init(id, ());
            }
        }
    }

    impl ServerDispatch<ServerWlBuffer, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlBuffer,
            _request: <ServerWlBuffer as Resource>::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl GlobalDispatch<ServerWlCompositor, ()> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerWlCompositor>,
            _global_data: &(),
            data_init: &mut DataInit<'_, Self>,
        ) {
            data_init.init(resource, ());
        }
    }

    impl ServerDispatch<ServerWlCompositor, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlCompositor,
            request: wl_compositor::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                wl_compositor::Request::CreateSurface { id } => {
                    let s = data_init.init(id, ());
                    let proto_id = s.id().protocol_id();
                    state.subsurface_stacks.insert(proto_id, vec![proto_id]);
                    state.surfaces.push(s);
                },
                wl_compositor::Request::CreateRegion { id } => {
                    data_init.init(id, ());
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerWlSurface, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            resource: &ServerWlSurface,
            request: server_surface::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            let surface_id = resource.id().protocol_id();
            match request {
                server_surface::Request::Attach { buffer, x, y } => {
                    state.xdg_reqs.push(HostXdgReq::SurfaceAttach {
                        surface_id,
                        has_buffer: buffer.is_some(),
                        x,
                        y,
                    });
                    if let Some(buffer) = buffer {
                        state.pending_buffers.insert(surface_id, buffer);
                    } else {
                        state.pending_buffers.remove(&surface_id);
                    }
                },
                server_surface::Request::Offset { x, y } => {
                    state
                        .xdg_reqs
                        .push(HostXdgReq::SurfaceOffset { surface_id, x, y });
                },
                server_surface::Request::DamageBuffer {
                    x,
                    y,
                    width,
                    height,
                } => {
                    state.damage_buffers.push((surface_id, x, y, width, height));
                },
                server_surface::Request::Frame { callback } => {
                    let cb = data_init.init(callback, ());
                    state.pending_frame_callbacks.push(cb);
                },
                server_surface::Request::Commit => {
                    state
                        .xdg_reqs
                        .push(HostXdgReq::SurfaceCommit { surface_id });
                    if let Some(buf) = state.pending_buffers.remove(&surface_id) {
                        if state.hold_committed_buffers {
                            // Hold the newly attached buffer and release any
                            // previously held buffer (modeling a real 2-slot
                            // Wayland compositor scanout swap).
                            for old in state.held_buffers.drain(..) {
                                old.release();
                            }
                            state.held_buffers.push(buf);
                        } else {
                            buf.release();
                        }
                    }
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerWlCallback, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlCallback,
            _request: <ServerWlCallback as Resource>::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl GlobalDispatch<ServerWlOutput, u32> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerWlOutput>,
            &tag: &u32,
            data_init: &mut DataInit<'_, Self>,
        ) {
            let output = data_init.init(resource, tag);
            output.geometry(
                (tag as i32 - 1) * 1920,
                0,
                600,
                340,
                server_output::Subpixel::HorizontalRgb,
                "Vendor".to_string(),
                format!("Model-{tag}"),
                server_output::Transform::Normal,
            );
            output.mode(
                server_output::Mode::Current | server_output::Mode::Preferred,
                1920,
                1080,
                60000,
            );
            output.scale(1);
            output.name(format!("OUT-{tag}"));
            output.done();
        }
    }

    impl ServerDispatch<ServerWlOutput, u32> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlOutput,
            _request: <ServerWlOutput as Resource>::Request,
            _data: &u32,
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl ServerDispatch<ServerWlRegion, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlRegion,
            _request: <ServerWlRegion as Resource>::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl GlobalDispatch<ServerWlSubcompositor, ()> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerWlSubcompositor>,
            _global_data: &(),
            data_init: &mut DataInit<'_, Self>,
        ) {
            data_init.init(resource, ());
        }
    }

    impl ServerDispatch<ServerWlSubcompositor, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlSubcompositor,
            request: wl_subcompositor::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            if let wl_subcompositor::Request::GetSubsurface {
                id,
                surface,
                parent,
            } = request
            {
                let surface_id = surface.id().protocol_id();
                let parent_id = parent.id().protocol_id();
                data_init.init(id, (surface_id, parent_id));
                state
                    .subsurface_stacks
                    .entry(parent_id)
                    .or_insert_with(|| vec![parent_id])
                    .push(surface_id);
            }
        }
    }

    impl ServerDispatch<ServerWlSubsurface, (u32, u32)> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlSubsurface,
            request: server_subsurface::Request,
            &(surface_id, parent_id): &(u32, u32),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_subsurface::Request::SetPosition { x, y } => {
                    state.subsurface_positions.insert(surface_id, (x, y));
                },
                server_subsurface::Request::PlaceAbove { sibling } => {
                    let sibling_id = sibling.id().protocol_id();
                    state.xdg_reqs.push(HostXdgReq::SubsurfacePlaceAbove {
                        surface_id,
                        sibling_id,
                    });
                    let stack = state.subsurface_stacks.get_mut(&parent_id).unwrap();
                    let src = stack.iter().position(|&id| id == surface_id).unwrap();
                    stack.remove(src);
                    let dst = stack.iter().position(|&id| id == sibling_id).unwrap();
                    stack.insert(dst + 1, surface_id);
                },
                server_subsurface::Request::PlaceBelow { sibling } => {
                    let sibling_id = sibling.id().protocol_id();
                    state.xdg_reqs.push(HostXdgReq::SubsurfacePlaceBelow {
                        surface_id,
                        sibling_id,
                    });
                    let stack = state.subsurface_stacks.get_mut(&parent_id).unwrap();
                    let src = stack.iter().position(|&id| id == surface_id).unwrap();
                    stack.remove(src);
                    let dst = stack.iter().position(|&id| id == sibling_id).unwrap();
                    stack.insert(dst, surface_id);
                },
                server_subsurface::Request::Destroy => {
                    if let Some(stack) = state.subsurface_stacks.get_mut(&parent_id) {
                        stack.retain(|&id| id != surface_id);
                    }
                },
                _ => {},
            }
        }
    }

    impl GlobalDispatch<ServerXdgWmBase, ()> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerXdgWmBase>,
            _global_data: &(),
            data_init: &mut DataInit<'_, Self>,
        ) {
            data_init.init(resource, ());
        }
    }

    impl ServerDispatch<ServerXdgWmBase, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerXdgWmBase,
            request: xdg_wm_base::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                xdg_wm_base::Request::CreatePositioner { id } => {
                    data_init.init(id, ());
                },
                xdg_wm_base::Request::GetXdgSurface { id, surface } => {
                    data_init.init(id, surface.id().protocol_id());
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerXdgPositioner, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerXdgPositioner,
            _request: <ServerXdgPositioner as Resource>::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl ServerDispatch<ServerXdgSurface, u32> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerXdgSurface,
            request: server_xdg_surface::Request,
            &surface_id: &u32,
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_xdg_surface::Request::GetToplevel { id } => {
                    data_init.init(id, surface_id);
                },
                server_xdg_surface::Request::GetPopup { id, .. } => {
                    let popup = data_init.init(id, surface_id);
                    state.xdg_popups.insert(surface_id, popup);
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerXdgToplevel, u32> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerXdgToplevel,
            request: server_xdg_toplevel::Request,
            &surface_id: &u32,
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_xdg_toplevel::Request::ShowWindowMenu { serial, x, y, .. } => {
                    state.xdg_reqs.push(HostXdgReq::ToplevelShowWindowMenu {
                        surface_id,
                        serial,
                        x,
                        y,
                    });
                },
                server_xdg_toplevel::Request::SetFullscreen { output } => {
                    let output_tag = output.as_ref().and_then(|o| o.data::<u32>().copied());
                    state.xdg_reqs.push(HostXdgReq::ToplevelSetFullscreen {
                        surface_id,
                        output_tag,
                    });
                },
                server_xdg_toplevel::Request::UnsetFullscreen => {
                    state
                        .xdg_reqs
                        .push(HostXdgReq::ToplevelUnsetFullscreen { surface_id });
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerXdgPopup, u32> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerXdgPopup,
            request: server_xdg_popup::Request,
            &surface_id: &u32,
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
            if let server_xdg_popup::Request::Grab { serial, .. } = request {
                state
                    .xdg_reqs
                    .push(HostXdgReq::PopupGrab { surface_id, serial });
            }
        }
    }

    impl GlobalDispatch<ServerWlSeat, ()> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerWlSeat>,
            _global_data: &(),
            data_init: &mut DataInit<'_, Self>,
        ) {
            let seat = data_init.init(resource, ());
            seat.name("seat0".to_string());
            seat.capabilities(server_seat::Capability::Pointer | server_seat::Capability::Keyboard);
        }
    }

    impl ServerDispatch<ServerWlSeat, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlSeat,
            request: server_seat::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_seat::Request::GetPointer { id } => {
                    state.pointer = Some(data_init.init(id, ()));
                },
                server_seat::Request::GetKeyboard { id } => {
                    state.keyboard = Some(data_init.init(id, ()));
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerWlKeyboard, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlKeyboard,
            _request: server_keyboard::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl ServerDispatch<ServerWlPointer, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlPointer,
            _request: server_pointer::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
        }
    }

    impl GlobalDispatch<ServerWlDataDeviceManager, ()> for MockHostState {
        fn bind(
            _state: &mut Self,
            _dh: &DisplayHandle,
            _client: &ServerClient,
            resource: New<ServerWlDataDeviceManager>,
            _global_data: &(),
            data_init: &mut DataInit<'_, Self>,
        ) {
            data_init.init(resource, ());
        }
    }

    impl ServerDispatch<ServerWlDataDeviceManager, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlDataDeviceManager,
            request: server_ddm::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_ddm::Request::CreateDataSource { id } => {
                    state.last_source = Some(data_init.init(id, ()));
                },
                server_ddm::Request::GetDataDevice { id, .. } => {
                    state.data_device = Some(data_init.init(id, ()));
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerWlDataSource, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlDataSource,
            request: server_data_source::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_data_source::Request::Offer { mime_type } => {
                    state.reqs.push(HostRecordedReq::SourceOffer(mime_type));
                },
                server_data_source::Request::SetActions {
                    dnd_actions: ServerWEnum::Value(actions),
                } => {
                    state.reqs.push(HostRecordedReq::SourceSetActions(actions));
                },
                server_data_source::Request::Destroy => {
                    state.reqs.push(HostRecordedReq::SourceDestroy);
                },
                _ => {},
            }
        }
    }

    impl ServerDispatch<ServerWlDataDevice, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlDataDevice,
            request: server_data_device::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
            if let server_data_device::Request::StartDrag {
                origin,
                icon,
                serial,
                ..
            } = request
            {
                state.reqs.push(HostRecordedReq::StartDrag {
                    origin_id: origin.id().protocol_id(),
                    has_icon: icon.is_some(),
                    serial,
                });
            }
        }
    }

    impl ServerDispatch<ServerWlDataOffer, ()> for MockHostState {
        fn request(
            state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlDataOffer,
            request: server_data_offer::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
            match request {
                server_data_offer::Request::Accept { serial, mime_type } => {
                    state
                        .reqs
                        .push(HostRecordedReq::OfferAccept { serial, mime_type });
                },
                server_data_offer::Request::SetActions {
                    dnd_actions: ServerWEnum::Value(dnd_actions),
                    preferred_action: ServerWEnum::Value(preferred_action),
                } => {
                    state.reqs.push(HostRecordedReq::OfferSetActions {
                        dnd_actions,
                        preferred_action,
                    });
                },
                server_data_offer::Request::Receive { mime_type, fd } => {
                    state
                        .reqs
                        .push(HostRecordedReq::OfferReceive(mime_type.clone()));
                    state.receive_fds.push((mime_type, fd));
                },
                server_data_offer::Request::Finish => {
                    state.reqs.push(HostRecordedReq::OfferFinish);
                },
                server_data_offer::Request::Destroy => {
                    state.reqs.push(HostRecordedReq::OfferDestroy);
                },
                _ => {},
            }
        }
    }

    struct ClientDndHarness {
        display: Display<MockHostState>,
        host: MockHostState,
        conn: Connection,
        event_queue: EventQueue<WprsClientState>,
        client: WprsClientState,
        rx: Receiver<SendType<Event>>,
    }

    impl ClientDndHarness {
        fn new() -> Self {
            Self::new_with_compositor_version(5).0
        }

        fn new_with_compositor_version(compositor_version: u32) -> (Self, HashMap<String, u32>) {
            let mut display: Display<MockHostState> = Display::new().unwrap();
            let mut dh = display.handle();
            dh.create_global::<MockHostState, ServerWlShm, _>(1, ());
            dh.create_global::<MockHostState, ServerWlCompositor, _>(compositor_version, ());
            dh.create_global::<MockHostState, ServerWlSubcompositor, _>(1, ());
            dh.create_global::<MockHostState, ServerXdgWmBase, _>(5, ());
            dh.create_global::<MockHostState, ServerWlDataDeviceManager, _>(3, ());
            dh.create_global::<MockHostState, ServerWlSeat, _>(7, ());
            dh.create_global::<MockHostState, ServerWlOutput, u32>(4, 1);
            dh.create_global::<MockHostState, ServerWlOutput, u32>(4, 2);

            let (server_sock, client_sock) = UnixStream::pair().unwrap();
            let server_client = dh
                .insert_client(server_sock, Arc::new(DummyClientData))
                .unwrap();
            let mut host = MockHostState {
                client: Some(server_client),
                ..Default::default()
            };

            let conn = Connection::from_socket(client_sock).unwrap();
            let done = Arc::new(AtomicBool::new(false));
            let done_bg = Arc::clone(&done);
            let bg = thread::spawn(move || {
                while !done_bg.load(Ordering::Acquire) {
                    display.dispatch_clients(&mut host).unwrap();
                    display.flush_clients().unwrap();
                    thread::sleep(Duration::from_millis(1));
                }
                display.dispatch_clients(&mut host).unwrap();
                display.flush_clients().unwrap();
                (display, host)
            });

            let (globals, event_queue) = registry_queue_init::<WprsClientState>(&conn).unwrap();
            done.store(true, Ordering::Release);
            let (display, host) = bg.join().unwrap();

            let (serializer, rx) = Serializer::new_test_client();
            let client = WprsClientState::new(
                event_queue.handle(),
                globals,
                conn.clone(),
                serializer,
                ClientOptions {
                    title_prefix: String::new(),
                    compression_mode: CompressionMode::Lossless,
                },
            )
            .unwrap();

            let mut harness = Self {
                display,
                host,
                conn,
                event_queue,
                client,
                rx,
            };
            // Pump seat capabilities -> pointer + data_device creation and initial outputs.
            harness.sync();
            harness.sync();
            let mut outputs_by_name = HashMap::new();
            for ev in harness.drain_events() {
                if let Event::Output(wayland::OutputEvent::New(info)) = ev
                    && let Some(name) = info.name
                {
                    outputs_by_name.insert(name, info.id);
                }
            }
            harness.host.reqs.clear();
            harness.host.xdg_reqs.clear();
            (harness, outputs_by_name)
        }

        fn sync(&mut self) {
            self.conn.flush().unwrap();
            self.display.dispatch_clients(&mut self.host).unwrap();
            self.display.flush_clients().unwrap();
            for _ in 0..4 {
                if let Some(guard) = self.conn.prepare_read() {
                    _ = guard.read();
                }
                self.event_queue.dispatch_pending(&mut self.client).unwrap();
                self.conn.flush().unwrap();
                self.display.dispatch_clients(&mut self.host).unwrap();
                self.display.flush_clients().unwrap();
            }
        }

        fn fire_frame_callbacks(&mut self) {
            for cb in self.host.pending_frame_callbacks.drain(..) {
                cb.done(0);
            }
            self.sync();
        }

        fn create_remote_surface(&mut self, cid: u64, sid: u64) -> ServerWlSurface {
            let client_id = ClientId(cid);
            let surface_id = WlSurfaceId(sid);
            let remote_client = self.client.remote_display.client(&client_id);
            let remote = RemoteSurface::new(
                client_id,
                surface_id,
                &self.client.compositor_state,
                &self.client.qh,
                &mut self.client.object_bimap,
            )
            .unwrap();
            let proto_id = remote.wl_surface().id().protocol_id();
            remote_client.surfaces.insert(surface_id, remote);
            self.sync();
            self.host
                .surfaces
                .iter()
                .find(|s| s.id().protocol_id() == proto_id)
                .expect("host must see created surface")
                .clone()
        }

        fn create_host_offer(&self) -> ServerWlDataOffer {
            let client = self.host.client.as_ref().unwrap();
            let offer = client
                .create_resource::<ServerWlDataOffer, _, MockHostState>(
                    &self.display.handle(),
                    3,
                    (),
                )
                .unwrap();
            self.host.data_device.as_ref().unwrap().data_offer(&offer);
            offer
        }

        fn drain_events(&self) -> Vec<Event> {
            let mut out = Vec::new();
            while let Ok(msg) = self.rx.try_recv() {
                if let SendType::Object(ev) = msg {
                    out.push(ev);
                }
            }
            out
        }

        fn recv_event_timeout(&self, timeout: Duration) -> Event {
            loop {
                match self
                    .rx
                    .recv_timeout(timeout)
                    .expect("timed out waiting for event")
                {
                    SendType::Object(ev) => return ev,
                    SendType::RawBuffer(_) => {},
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_dnd_offer_leave_and_post_drop_ask() {
        let mut h = ClientDndHarness::new();
        let surf = h.create_remote_surface(1, 10);

        // 1. Enter -> Motion -> Leave (no drop) must emit DnDEnter, DnDMotion,
        // DnDLeave and clear `active_dnd_offer` (fixes #148 leave no-op).
        let offer1 = h.create_host_offer();
        offer1.offer("text/plain".to_string());
        offer1.offer("_wprs_marker".to_string());
        offer1.source_actions(ServerDndAction::Copy | ServerDndAction::Move | ServerDndAction::Ask);
        h.host
            .data_device
            .as_ref()
            .unwrap()
            .enter(777, &surf, 12.0, 34.0, Some(&offer1));
        h.host.data_device.as_ref().unwrap().motion(1, 20.0, 30.0);
        h.sync();

        let events = h.drain_events();
        let offer_id_1 = events
            .iter()
            .find_map(|e| match e {
                Event::Data(DataEvent::DestinationEvent(DataDestinationEvent::DnDEnter(enter))) => {
                    assert_eq!(enter.serial, 777);
                    assert_eq!(enter.surface, WlSurfaceId(10));
                    assert_eq!(enter.loc, (12.0, 34.0).into());
                    Some(enter.offer_id)
                },
                _ => None,
            })
            .expect("wprsc must emit DnDEnter without skipping drags");
        assert!(events.contains(&Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDMotion {
                offer_id: offer_id_1,
                loc: (20.0, 30.0).into(),
            },
        ))));

        h.host.data_device.as_ref().unwrap().leave();
        h.sync();
        assert_eq!(
            h.drain_events(),
            vec![Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::DnDLeave {
                    offer_id: offer_id_1,
                },
            ))]
        );
        assert!(h.client.active_dnd_offer.is_none());

        // 2. Drop followed by leave preserves the offer in `dropped_dnd_offers`
        // so post-drop Ask accept/set_actions/receive/finish all reach the host
        // using the real enter serial (bypassing SCTK's `self.left` drop).
        let offer2 = h.create_host_offer();
        offer2.offer("text/plain".to_string());
        offer2.source_actions(ServerDndAction::Copy | ServerDndAction::Move | ServerDndAction::Ask);
        let dd = h.host.data_device.as_ref().unwrap().clone();
        dd.enter(888, &surf, 5.0, 6.0, Some(&offer2));
        dd.drop();
        dd.leave();
        h.sync();

        let events = h.drain_events();
        let offer_id_2 = events
            .iter()
            .find_map(|e| match e {
                Event::Data(DataEvent::DestinationEvent(DataDestinationEvent::DnDEnter(enter))) => {
                    Some(enter.offer_id)
                },
                _ => None,
            })
            .unwrap();
        assert!(events.contains(&Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDDrop {
                offer_id: offer_id_2,
            },
        ))));
        assert!(!events.iter().any(|e| matches!(
            e,
            Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::DnDLeave { offer_id }
            )) if *offer_id == offer_id_2
        )));
        h.host.reqs.clear();

        // Post-drop accept + set_actions + receive + finish.
        h.client.handle_request(RecvType::Object(Request::Data(
            DataRequest::DestinationRequest(DataDestinationRequest::DnDAcceptMimeType {
                offer_id: offer_id_2,
                mime_type: Some("text/plain".to_string()),
            }),
        )));
        h.client.handle_request(RecvType::Object(Request::Data(
            DataRequest::DestinationRequest(DataDestinationRequest::DnDSetDestinationActions {
                offer_id: offer_id_2,
                dnd_actions: ServerDndAction::Move.into(),
                preferred_action: ServerDndAction::Move.into(),
            }),
        )));
        h.client.handle_request(RecvType::Object(Request::Data(
            DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer {
                target: DataTransferTarget::DnD(offer_id_2),
                mime_type: "text/plain".to_string(),
                transfer_id: 42,
            }),
        )));
        h.sync();

        assert_eq!(h.host.receive_fds.len(), 1);
        let (mime, fd) = h.host.receive_fds.pop().unwrap();
        assert_eq!(mime, "text/plain");
        File::from(fd).write_all(b"post-drop-payload").unwrap();
        assert_eq!(
            h.recv_event_timeout(Duration::from_secs(2)),
            Event::Data(DataEvent::TransferData(
                42,
                DataToTransfer(b"post-drop-payload".to_vec()),
            ))
        );

        h.client.handle_request(RecvType::Object(Request::Data(
            DataRequest::DestinationRequest(DataDestinationRequest::DnDFinish {
                offer_id: offer_id_2,
            }),
        )));
        h.sync();
        assert_eq!(
            h.host.reqs,
            vec![
                HostRecordedReq::OfferAccept {
                    serial: 888,
                    mime_type: Some("text/plain".to_string()),
                },
                HostRecordedReq::OfferSetActions {
                    dnd_actions: ServerDndAction::Move,
                    preferred_action: ServerDndAction::Move,
                },
                HostRecordedReq::OfferReceive("text/plain".to_string()),
                HostRecordedReq::OfferFinish,
                HostRecordedReq::OfferDestroy,
            ]
        );

        // 3. Dropped Ask offer cancelled via DnDCancel destroys host offer
        // without calling finish().
        let offer3 = h.create_host_offer();
        offer3.offer("text/uri-list".to_string());
        dd.enter(999, &surf, 1.0, 2.0, Some(&offer3));
        dd.drop();
        dd.leave();
        h.sync();
        let offer_id_3 = h
            .drain_events()
            .into_iter()
            .find_map(|e| match e {
                Event::Data(DataEvent::DestinationEvent(DataDestinationEvent::DnDEnter(enter))) => {
                    Some(enter.offer_id)
                },
                _ => None,
            })
            .unwrap();
        h.host.reqs.clear();
        h.client.handle_request(RecvType::Object(Request::Data(
            DataRequest::DestinationRequest(DataDestinationRequest::DnDCancel {
                offer_id: offer_id_3,
            }),
        )));
        h.sync();
        assert_eq!(h.host.reqs, vec![HostRecordedReq::OfferDestroy]);

        // 4. RequestDataTransfer for nonexistent offer replies with empty bytes.
        h.client.handle_request(RecvType::Object(Request::Data(
            DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer {
                target: DataTransferTarget::DnD(9999),
                mime_type: "text/plain".to_string(),
                transfer_id: 43,
            }),
        )));
        assert_eq!(
            h.drain_events(),
            vec![Event::Data(DataEvent::TransferData(
                43,
                DataToTransfer(Vec::new()),
            ))]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::verbose_file_reads)]
    fn test_wprsc_start_drag_origin_subsurface_leave_and_silent_host_reject() {
        let mut h = ClientDndHarness::new();
        let surf_a = h.create_remote_surface(1, 10);
        let surf_b = h.create_remote_surface(1, 11);
        let ptr = h.host.pointer.as_ref().unwrap().clone();

        // Press BTN_LEFT on surf_a (with NO keyboard focus), then cross onto
        // surf_b in the same pointer frame: pressed_buttons must survive.
        ptr.enter(10, &surf_a, 5.0, 5.0);
        ptr.button(11, 1, 0x110, server_pointer::ButtonState::Pressed);
        ptr.frame();
        ptr.leave(12, &surf_a);
        ptr.enter(13, &surf_b, 2.0, 2.0);
        ptr.frame();
        h.sync();
        h.drain_events();
        h.host.reqs.clear();

        // StartDrag targets origin surf_a with icon and explicit serial 11, and
        // must NOT append `_wprs_marker`.
        h.client
            .handle_request(RecvType::Object(Request::Data(DataRequest::SourceRequest(
                DataSourceRequest::StartDrag(StartDrag {
                    source_id: 5,
                    source: SourceMetadata {
                        mime_types: vec!["text/plain".to_string()],
                        dnd_actions: (ServerDndAction::Copy | ServerDndAction::Move).into(),
                    },
                    origin: ClientSurface {
                        client: ClientId(1),
                        surface: WlSurfaceId(10),
                    },
                    icon: Some(ClientSurface {
                        client: ClientId(1),
                        surface: WlSurfaceId(12),
                    }),
                    serial: 11,
                }),
            ))));
        h.sync();
        assert_eq!(
            h.host.reqs,
            vec![
                HostRecordedReq::SourceOffer("text/plain".to_string()),
                HostRecordedReq::SourceSetActions(ServerDndAction::Copy | ServerDndAction::Move),
                HostRecordedReq::StartDrag {
                    origin_id: surf_a.id().protocol_id(),
                    has_icon: true,
                    serial: 11,
                },
            ]
        );
        h.host.reqs.clear();

        // Simulate Smithay#1887 host compositor silently rejecting StartDrag
        // without sending `wl_data_source.cancelled`, and instead delivering
        // BTN_LEFT release: wprsc must synthesize DnDCancelled { source_id: 5 }.
        ptr.button(14, 2, 0x110, server_pointer::ButtonState::Released);
        ptr.frame();
        h.sync();
        let events = h.drain_events();
        assert!(events.contains(&Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDCancelled { source_id: 5 },
        ))));
        assert!(h.host.reqs.contains(&HostRecordedReq::SourceDestroy));
        h.host.reqs.clear();

        // With pressed_buttons empty, a late StartDrag is cancelled immediately
        // without calling start_drag on the host.
        h.client
            .handle_request(RecvType::Object(Request::Data(DataRequest::SourceRequest(
                DataSourceRequest::StartDrag(StartDrag {
                    source_id: 6,
                    source: SourceMetadata {
                        mime_types: vec!["text/plain".to_string()],
                        dnd_actions: ServerDndAction::Copy.into(),
                    },
                    origin: ClientSurface {
                        client: ClientId(1),
                        surface: WlSurfaceId(10),
                    },
                    icon: None,
                    serial: 11,
                }),
            ))));
        h.sync();
        assert_eq!(
            h.drain_events(),
            vec![Event::Data(DataEvent::SourceEvent(
                DataSourceEvent::DnDCancelled { source_id: 6 },
            ))]
        );
        assert!(h.host.reqs.is_empty());

        // Accepted StartDrag forwards target, action, drop_performed, send
        // (with TransferData write-back), and dnd_finished tagged by source_id.
        ptr.button(20, 3, 0x110, server_pointer::ButtonState::Pressed);
        ptr.frame();
        h.sync();
        h.drain_events();

        h.client
            .handle_request(RecvType::Object(Request::Data(DataRequest::SourceRequest(
                DataSourceRequest::StartDrag(StartDrag {
                    source_id: 7,
                    source: SourceMetadata {
                        mime_types: vec!["text/plain".to_string()],
                        dnd_actions: ServerDndAction::Move.into(),
                    },
                    origin: ClientSurface {
                        client: ClientId(1),
                        surface: WlSurfaceId(10),
                    },
                    icon: None,
                    serial: 20,
                }),
            ))));
        h.sync();

        let host_source = h.host.last_source.clone().unwrap();
        let (r_fd, w_fd) = unistd::pipe2(OFlag::O_CLOEXEC).unwrap();
        host_source.target(Some("text/plain".to_string()));
        host_source.action(ServerDndAction::Move);
        host_source.dnd_drop_performed();
        host_source.send("text/plain".to_string(), w_fd.as_fd());
        drop(w_fd);
        host_source.dnd_finished();
        h.sync();

        let events = h.drain_events();
        assert!(events.contains(&Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDMimeTypeAcceptedByDestination {
                source_id: 7,
                mime_type: Some("text/plain".to_string()),
            },
        ))));
        assert!(events.contains(&Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDActionSelected {
                source_id: 7,
                action: ServerDndAction::Move.into(),
            },
        ))));
        assert!(events.contains(&Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDDropPerformed { source_id: 7 },
        ))));
        assert!(events.contains(&Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDFinished { source_id: 7 },
        ))));
        let transfer_id = events
            .iter()
            .find_map(|e| match e {
                Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::MimeTypeSendRequestedByDestination {
                        target: DataTransferTarget::DnD(7),
                        mime_type,
                        transfer_id,
                    },
                )) => {
                    assert_eq!(mime_type, "text/plain");
                    Some(*transfer_id)
                },
                _ => None,
            })
            .expect("wprsc must forward send_request with source_id 7");

        h.client
            .handle_request(RecvType::Object(Request::Data(DataRequest::TransferData(
                transfer_id,
                DataToTransfer(b"remote-src-bytes".to_vec()),
            ))));
        let mut received = Vec::new();
        File::from(r_fd).read_to_end(&mut received).unwrap();
        assert_eq!(received, b"remote-src-bytes");
    }

    fn make_client_surface_state(
        client: ClientId,
        id: WlSurfaceId,
        role: Option<wayland::Role>,
    ) -> SurfaceState {
        SurfaceState {
            client,
            id,
            buffer: None,
            role,
            buffer_delta: None,
            buffer_scale: 1,
            buffer_transform: None,
            opaque_region: None,
            input_region: None,
            z_ordered_children: vec![wayland::SubsurfacePosition {
                id,
                position: (0, 0).into(),
            }],
            damage: None,
            output_ids: Vec::new(),
            viewport_state: None,
            xdg_surface_state: None,
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_popup_grab_ordering_skip_without_serial_and_done_wire() {
        let mut h = ClientDndHarness::new();
        let client_id = ClientId(1);
        let parent_id = WlSurfaceId(100);
        let popup_grab_id = WlSurfaceId(101);
        let popup_nograb_id = WlSurfaceId(102);

        // Create parent toplevel first.
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: parent_id,
                payload: SurfaceRequestPayload::Commit(make_client_surface_state(
                    client_id,
                    parent_id,
                    Some(wayland::Role::XdgToplevel(Default::default())),
                )),
            })));
        h.sync();
        h.host.xdg_reqs.clear();
        h.drain_events();

        let positioner = xdg_shell::XdgPositioner {
            width: 120,
            height: 80,
            anchor_rect: crate::serialization::geometry::Rectangle::new(10, 10, 1, 1),
            anchor_edges: 5, // TopLeft
            gravity: 8,      // BottomRight
            constraint_adjustment: 0,
            offset: (0, 0).into(),
            reactive: false,
            parent_size: None,
            parent_configure: None,
        };

        // 1. Popup with `grab_serial: Some(4242)` must issue `xdg_popup.grab`
        //    BEFORE its initial `wl_surface.commit()`.
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: popup_grab_id,
                payload: SurfaceRequestPayload::Commit(make_client_surface_state(
                    client_id,
                    popup_grab_id,
                    Some(wayland::Role::XdgPopup(xdg_shell::XdgPopupState {
                        parent_surface_id: parent_id,
                        positioner,
                        grab_serial: Some(4242),
                        reposition_token: None,
                    })),
                )),
            })));
        h.sync();

        let grab_proto_id = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&popup_grab_id)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        assert_eq!(
            h.host.xdg_reqs,
            vec![
                HostXdgReq::PopupGrab {
                    surface_id: grab_proto_id,
                    serial: 4242,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: grab_proto_id,
                },
            ]
        );

        // Host rejects or dismisses the grab -> `wprsc` emits `PopupEvent::Done`.
        h.host.xdg_popups[&grab_proto_id].popup_done();
        h.sync();
        assert_eq!(
            h.drain_events(),
            vec![Event::Popup(xdg_shell::PopupEvent::Done(popup_grab_id))]
        );

        // 2. Popup with `grab_serial: None` commits without calling `grab`.
        h.host.xdg_reqs.clear();
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: popup_nograb_id,
                payload: SurfaceRequestPayload::Commit(make_client_surface_state(
                    client_id,
                    popup_nograb_id,
                    Some(wayland::Role::XdgPopup(xdg_shell::XdgPopupState {
                        parent_surface_id: parent_id,
                        positioner,
                        grab_serial: None,
                        reposition_token: None,
                    })),
                )),
            })));
        h.sync();

        let nograb_proto_id = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&popup_nograb_id)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        assert_eq!(
            h.host.xdg_reqs,
            vec![HostXdgReq::SurfaceCommit {
                surface_id: nograb_proto_id,
            }]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_subsurface_z_order_above_below_parent_and_nested_wire() {
        let mut h = ClientDndHarness::new();
        let client_id = ClientId(1);
        let p = WlSurfaceId(200);
        let c1 = WlSurfaceId(201);
        let c2 = WlSurfaceId(202);
        let c3 = WlSurfaceId(203);
        let gc = WlSurfaceId(204);
        let sp = |id, x, y| wayland::SubsurfacePosition {
            id,
            position: (x, y).into(),
        };

        let proto_id_of = |h: &mut ClientDndHarness, sid: WlSurfaceId| {
            h.client
                .remote_display
                .client(&client_id)
                .surfaces
                .get(&sid)
                .unwrap()
                .wl_surface()
                .id()
                .protocol_id()
        };

        // 1. Initial commit places `c2` and `c1` below `p` and `c3` above `p`:
        //    bottom-to-top `[c2, c1, p, c3]`.
        let mut p_state = make_client_surface_state(
            client_id,
            p,
            Some(wayland::Role::XdgToplevel(Default::default())),
        );
        p_state.z_ordered_children =
            vec![sp(c2, 30, 40), sp(c1, 10, 20), sp(p, 0, 0), sp(c3, 50, 60)];
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: p,
                payload: SurfaceRequestPayload::Commit(p_state),
            })));
        h.sync();

        let p_proto = proto_id_of(&mut h, p);
        let c1_proto = proto_id_of(&mut h, c1);
        let c2_proto = proto_id_of(&mut h, c2);
        let c3_proto = proto_id_of(&mut h, c3);
        assert_eq!(
            h.host.subsurface_stacks[&p_proto],
            vec![c2_proto, c1_proto, p_proto, c3_proto]
        );
        assert!(h.host.xdg_reqs.contains(&HostXdgReq::SubsurfacePlaceBelow {
            surface_id: c1_proto,
            sibling_id: p_proto,
        }));
        assert!(h.host.xdg_reqs.contains(&HostXdgReq::SubsurfacePlaceBelow {
            surface_id: c2_proto,
            sibling_id: c1_proto,
        }));
        assert_eq!(h.host.subsurface_positions[&c1_proto], (10, 20));
        assert_eq!(h.host.subsurface_positions[&c2_proto], (30, 40));
        assert_eq!(h.host.subsurface_positions[&c3_proto], (50, 60));

        // 2. Nested grandchild `gc` below `c1` updates `c1`'s stack without
        //    altering `p`'s direct stack.
        h.host.xdg_reqs.clear();
        let mut c1_state = make_client_surface_state(
            client_id,
            c1,
            Some(wayland::Role::SubSurface(wayland::SubSurfaceState {
                parent: p,
                location: (10, 20).into(),
                sync: true,
            })),
        );
        c1_state.z_ordered_children = vec![sp(gc, 7, 8), sp(c1, 0, 0)];
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: c1,
                payload: SurfaceRequestPayload::Commit(c1_state),
            })));
        h.sync();

        let gc_proto = proto_id_of(&mut h, gc);
        assert_eq!(
            h.host.subsurface_stacks[&c1_proto],
            vec![gc_proto, c1_proto]
        );
        assert_eq!(
            h.host.subsurface_stacks[&p_proto],
            vec![c2_proto, c1_proto, p_proto, c3_proto]
        );

        // 3. Reorder `p`'s children across `p` to `[c3, p, c2, c1]`.
        h.host.xdg_reqs.clear();
        let mut p_state2 = make_client_surface_state(
            client_id,
            p,
            Some(wayland::Role::XdgToplevel(Default::default())),
        );
        p_state2.z_ordered_children =
            vec![sp(c3, 50, 60), sp(p, 0, 0), sp(c2, 30, 40), sp(c1, 10, 20)];
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: p,
                payload: SurfaceRequestPayload::Commit(p_state2.clone()),
            })));
        h.sync();
        assert_eq!(
            h.host.subsurface_stacks[&p_proto],
            vec![c3_proto, p_proto, c2_proto, c1_proto]
        );

        // 4. Re-committing unchanged order sends zero place_above/place_below requests.
        h.host.xdg_reqs.clear();
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: p,
                payload: SurfaceRequestPayload::Commit(p_state2),
            })));
        h.sync();
        assert!(!h.host.xdg_reqs.iter().any(|r| matches!(
            r,
            HostXdgReq::SubsurfacePlaceAbove { .. } | HostXdgReq::SubsurfacePlaceBelow { .. }
        )));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_toplevel_show_window_menu_wire() {
        let mut h = ClientDndHarness::new();
        let client_id = ClientId(1);
        let tl_id = WlSurfaceId(300);

        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: tl_id,
                payload: SurfaceRequestPayload::Commit(make_client_surface_state(
                    client_id,
                    tl_id,
                    Some(wayland::Role::XdgToplevel(Default::default())),
                )),
            })));
        h.sync();
        h.host.xdg_reqs.clear();

        h.client
            .handle_request(RecvType::Object(Request::Toplevel(ToplevelRequest {
                client: client_id,
                surface: tl_id,
                payload: ToplevelRequestPayload::ShowWindowMenu {
                    serial: 777,
                    location: (120, 45).into(),
                },
            })));
        h.sync();

        let tl_proto = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&tl_id)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        assert_eq!(
            h.host.xdg_reqs,
            vec![HostXdgReq::ToplevelShowWindowMenu {
                surface_id: tl_proto,
                serial: 777,
                x: 120,
                y: 45,
            }]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_buffer_delta_v5_offset_and_v4_attach_wire() {
        use std::num::NonZeroUsize;

        use crate::buffer_pointer::BufferPointer;
        use crate::serialization::geometry::Point;
        use crate::serialization::wayland::BufferFormat;
        use crate::serialization::wayland::BufferMetadata;
        use crate::sharding_compression::ShardingCompressor;

        let send_full_buffer = |h: &mut ClientDndHarness,
                                client_id: ClientId,
                                surface_id: WlSurfaceId,
                                fill: u8,
                                delta: Option<Point<i32>>| {
            let meta = BufferMetadata {
                width: 4,
                height: 4,
                stride: 16,
                format: BufferFormat::Argb8888,
            };
            let pixels = [fill; 64];
            let ptr = pixels.as_ptr();
            // SAFETY: `pixels` stays alive for the duration of `filter_and_compress_full_streaming`.
            let bp = unsafe { BufferPointer::new(&ptr, pixels.len()) };
            let mut compressor = ShardingCompressor::new(NonZeroUsize::new(1).unwrap(), 1).unwrap();
            let mut streams = Vec::new();
            filtering::filter_and_compress_full_streaming(
                bp,
                client_id,
                surface_id,
                meta,
                CompressionMode::Lossless,
                None,
                None,
                &mut compressor,
                |stream| streams.push(stream),
            );
            for stream in streams {
                let (tx, rx) = crossbeam_channel::unbounded();
                for tile in stream.tiles {
                    tx.send(Ok(tile)).unwrap();
                }
                drop(tx);
                h.client
                    .handle_request(RecvType::RawBuffer(CompressedTileStream {
                        header: stream.header,
                        tiles: rx,
                    }));
            }
            let mut st = make_client_surface_state(client_id, surface_id, None);
            st.buffer = Some(BufferAssignment::New(meta));
            st.buffer_delta = delta;
            h.client
                .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                    client: client_id,
                    surface: surface_id,
                    payload: SurfaceRequestPayload::Commit(st),
                })));
            h.sync();
        };

        // 1. Host with `wl_compositor` v5 uses `attach(buf, 0, 0)` + `wl_surface.offset(dx, dy)`
        //    and does not replay the delta on subsequent delta-less commits.
        let (mut h5, _) = ClientDndHarness::new_with_compositor_version(5);
        let client_id = ClientId(1);
        let sid = WlSurfaceId(400);
        send_full_buffer(&mut h5, client_id, sid, 0x11, Some((-12, 7).into()));
        let proto_v5 = h5
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&sid)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        assert_eq!(
            h5.host.xdg_reqs,
            vec![
                HostXdgReq::SurfaceAttach {
                    surface_id: proto_v5,
                    has_buffer: true,
                    x: 0,
                    y: 0,
                },
                HostXdgReq::SurfaceOffset {
                    surface_id: proto_v5,
                    x: -12,
                    y: 7,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: proto_v5,
                },
            ]
        );

        // Delta-less next commit emits no `SurfaceOffset`.
        h5.fire_frame_callbacks();
        h5.host.xdg_reqs.clear();
        send_full_buffer(&mut h5, client_id, sid, 0x22, None);
        assert_eq!(
            h5.host.xdg_reqs,
            vec![
                HostXdgReq::SurfaceAttach {
                    surface_id: proto_v5,
                    has_buffer: true,
                    x: 0,
                    y: 0,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: proto_v5,
                },
            ]
        );

        // Commits arriving while throttled accumulate relative deltas and apply
        // them once on the next frame callback commit.
        h5.host.xdg_reqs.clear();
        for delta in [(-3, 2), (-5, 4)] {
            let mut throttled = make_client_surface_state(client_id, sid, None);
            throttled.buffer_delta = Some(delta.into());
            h5.client
                .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                    client: client_id,
                    surface: sid,
                    payload: SurfaceRequestPayload::Commit(throttled),
                })));
        }
        h5.sync();
        assert!(h5.host.xdg_reqs.is_empty());

        h5.fire_frame_callbacks();
        assert_eq!(
            h5.host.xdg_reqs,
            vec![
                HostXdgReq::SurfaceOffset {
                    surface_id: proto_v5,
                    x: -8,
                    y: 6,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: proto_v5,
                },
            ]
        );

        // Bufferless offset-only commit on v5 still emits `wl_surface.offset`.
        h5.host.xdg_reqs.clear();
        let mut offset_only = make_client_surface_state(client_id, sid, None);
        offset_only.buffer_delta = Some((5, -2).into());
        h5.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: sid,
                payload: SurfaceRequestPayload::Commit(offset_only),
            })));
        h5.sync();
        assert_eq!(
            h5.host.xdg_reqs,
            vec![
                HostXdgReq::SurfaceOffset {
                    surface_id: proto_v5,
                    x: 5,
                    y: -2,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: proto_v5,
                },
            ]
        );

        // 2. Host with `wl_compositor` v4 passes `(dx, dy)` via `wl_surface.attach`
        //    and never sends `wl_surface.offset`.
        let (mut h4, _) = ClientDndHarness::new_with_compositor_version(4);
        send_full_buffer(&mut h4, client_id, sid, 0x33, Some((-8, 3).into()));
        let proto_v4 = h4
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&sid)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        assert_eq!(
            h4.host.xdg_reqs,
            vec![
                HostXdgReq::SurfaceAttach {
                    surface_id: proto_v4,
                    has_buffer: true,
                    x: -8,
                    y: 3,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: proto_v4,
                },
            ]
        );

        h4.fire_frame_callbacks();
        h4.host.xdg_reqs.clear();
        send_full_buffer(&mut h4, client_id, sid, 0x44, None);
        assert_eq!(
            h4.host.xdg_reqs,
            vec![
                HostXdgReq::SurfaceAttach {
                    surface_id: proto_v4,
                    has_buffer: true,
                    x: 0,
                    y: 0,
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: proto_v4,
                },
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_toplevel_set_fullscreen_output_and_unset_wire() {
        let (mut h, outputs_by_name) = ClientDndHarness::new_with_compositor_version(5);
        let out1_id = outputs_by_name["OUT-1"];
        let out2_id = outputs_by_name["OUT-2"];
        let client_id = ClientId(1);
        let tl_id = WlSurfaceId(500);

        // 1. Pre-commit fullscreen on OUT-2 resolves to host output tag 2 in `set_role`.
        let init_tl = xdg_shell::XdgToplevelState {
            fullscreen: Some(xdg_shell::FullscreenState::Set(Some(out2_id))),
            ..Default::default()
        };
        h.client
            .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                client: client_id,
                surface: tl_id,
                payload: SurfaceRequestPayload::Commit(make_client_surface_state(
                    client_id,
                    tl_id,
                    Some(wayland::Role::XdgToplevel(init_tl)),
                )),
            })));
        h.sync();

        let tl_proto = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&tl_id)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        assert_eq!(
            h.host.xdg_reqs,
            vec![
                HostXdgReq::ToplevelSetFullscreen {
                    surface_id: tl_proto,
                    output_tag: Some(2),
                },
                HostXdgReq::SurfaceCommit {
                    surface_id: tl_proto,
                },
            ]
        );

        // 2. Post-commit `SetFullscreen(Some(out1_id))`, `SetFullscreen(None)`,
        //    unknown output fallback, and `UnsetFullscreen`.
        h.host.xdg_reqs.clear();
        for payload in [
            ToplevelRequestPayload::SetFullscreen(Some(out1_id)),
            ToplevelRequestPayload::SetFullscreen(None),
            ToplevelRequestPayload::SetFullscreen(Some(999_999)),
            ToplevelRequestPayload::UnsetFullscreen,
        ] {
            h.client
                .handle_request(RecvType::Object(Request::Toplevel(ToplevelRequest {
                    client: client_id,
                    surface: tl_id,
                    payload,
                })));
        }
        h.sync();
        assert_eq!(
            h.host.xdg_reqs,
            vec![
                HostXdgReq::ToplevelSetFullscreen {
                    surface_id: tl_proto,
                    output_tag: Some(1),
                },
                HostXdgReq::ToplevelSetFullscreen {
                    surface_id: tl_proto,
                    output_tag: None,
                },
                HostXdgReq::ToplevelSetFullscreen {
                    surface_id: tl_proto,
                    output_tag: None,
                },
                HostXdgReq::ToplevelUnsetFullscreen {
                    surface_id: tl_proto,
                },
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_frame_ack_no_leak_on_valid_unknown_and_corrupt_buffer() {
        use std::num::NonZeroUsize;

        use crate::buffer_pointer::BufferPointer;
        use crate::serialization::framing::CompressedTile;
        use crate::serialization::framing::LeafPayload;
        use crate::serialization::framing::RawBufferHeader;
        use crate::serialization::framing::TilePayload;
        use crate::serialization::framing::TileRect;
        use crate::serialization::wayland::BufferFormat;
        use crate::serialization::wayland::BufferMetadata;
        use crate::sharding_compression::ShardingCompressor;

        let mut h = ClientDndHarness::new();
        let client_id = ClientId(7);
        let sid_valid = WlSurfaceId(701);
        let sid_corrupt = WlSurfaceId(702);

        // 1. Valid RawBuffer for a previously unknown surface emits `Event::FrameAck { seq: 11 }`.
        let meta = BufferMetadata {
            width: 4,
            height: 4,
            stride: 16,
            format: BufferFormat::Argb8888,
        };
        let pixels = [0x42u8; 64];
        let ptr = pixels.as_ptr();
        // SAFETY: `pixels` stays alive for the duration of `filter_and_compress_full_streaming`.
        let bp = unsafe { BufferPointer::new(&ptr, pixels.len()) };
        let mut compressor = ShardingCompressor::new(NonZeroUsize::new(1).unwrap(), 1).unwrap();
        let mut streams = Vec::new();
        filtering::filter_and_compress_full_streaming(
            bp,
            client_id,
            sid_valid,
            meta,
            CompressionMode::Lossless,
            None,
            None,
            &mut compressor,
            |mut stream| {
                stream.header.frame_seq = 11;
                streams.push(stream);
            },
        );
        assert_eq!(streams.len(), 1);
        let stream = streams.pop().unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        for tile in stream.tiles {
            tx.send(Ok(tile)).unwrap();
        }
        drop(tx);
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: stream.header,
                tiles: rx,
            }));
        assert_eq!(h.drain_events(), vec![Event::FrameAck { seq: 11 }]);

        // 2. Corrupted/invalid RawBuffer emits `BufferResyncRequested` BEFORE
        //    `Event::FrameAck { seq: 12 }` so credits never leak and any deferred
        //    flush on `wprsd` sends a full frame.
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(Ok(CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::Solid([0xDE, 0xAD, 0xBE, 0xFF])),
        }))
        .unwrap();
        drop(tx);
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 12,
                    client: client_id,
                    surface: sid_corrupt,
                    metadata: meta,
                    incremental: false,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 99,
                        h: 99,
                    }],
                },
                tiles: rx,
            }));
        assert_eq!(
            h.drain_events(),
            vec![
                Event::Surface(SurfaceEvent {
                    surface_id: sid_corrupt,
                    payload: SurfaceEventPayload::BufferResyncRequested,
                }),
                Event::FrameAck { seq: 12 },
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_heartbeat_stale_input_suppression_and_timeout_wire() {
        use std::time::Instant;

        use smithay_client_toolkit::seat::keyboard::KeyEvent;
        use smithay_client_toolkit::seat::keyboard::KeyboardHandler;
        use smithay_client_toolkit::seat::keyboard::Keysym;

        use crate::client::HeartbeatTickAction;
        use crate::constants;
        use crate::serialization::wayland::KeyInner;
        use crate::serialization::wayland::KeyState;
        use crate::serialization::wayland::KeyboardEvent;
        use crate::serialization::wayland::PointerEventKind;

        let mut h = ClientDndHarness::new();
        let host_surf = h.create_remote_surface(1, 800);
        let sid = WlSurfaceId(800);
        let client_surf = h
            .client
            .remote_display
            .client(&ClientId(1))
            .surfaces
            .get(&sid)
            .unwrap()
            .wl_surface()
            .clone();
        let client_kbd = h.client.seat_objects[0]
            .keyboard
            .as_ref()
            .expect("seat must have keyboard")
            .clone();
        let host_ptr = h.host.pointer.clone().expect("host must have pointer");
        let qh = h.event_queue.handle();

        let make_key = |raw_code: u32| KeyEvent {
            time: 1,
            raw_code,
            keysym: Keysym::new(raw_code),
            utf8: None,
        };

        // 1. Pre-stall: enter, press key 30 and pointer button 0x110 -> forwarded.
        h.client
            .enter(&h.conn, &qh, &client_kbd, &client_surf, 1, &[], &[]);
        h.client
            .press_key(&h.conn, &qh, &client_kbd, 2, make_key(30));
        host_ptr.enter(3, &host_surf, 10.0, 20.0);
        host_ptr.button(4, 1, 0x110, server_pointer::ButtonState::Pressed);
        host_ptr.frame();
        h.sync();

        let pre_events = h.drain_events();
        assert!(pre_events.iter().any(|e| matches!(
            e,
            Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                raw_code: 30,
                state: KeyState::Pressed,
                ..
            }))
        )));
        assert!(pre_events.iter().any(|e| matches!(
            e,
            Event::PointerFrame(frames) if frames.iter().any(|p| matches!(
                p.kind,
                PointerEventKind::Press { button: 0x110, .. }
            ))
        )));

        // 2. Send a heartbeat stamped `STALE_INPUT_THRESHOLD + 1s` in the past so
        //    input is immediately considered stalled without sleeping.
        let stalled_sent_at =
            Instant::now() - constants::STALE_INPUT_THRESHOLD - Duration::from_secs(1);
        assert_eq!(
            h.client.handle_heartbeat_tick(stalled_sent_at),
            HeartbeatTickAction::Continue
        );
        assert_eq!(h.drain_events(), vec![Event::Heartbeat { seq: 1 }]);

        // 3. While stalled: blind typing (key 31 press/repeat/release) and blind
        //    pointer motion/click/scroll are suppressed, while releasing the
        //    pre-stall held key 30 and button 0x110 STILL forwards.
        h.client
            .press_key(&h.conn, &qh, &client_kbd, 5, make_key(31));
        h.client
            .repeat_key(&h.conn, &qh, &client_kbd, 6, make_key(31));
        h.client
            .release_key(&h.conn, &qh, &client_kbd, 7, make_key(31));
        h.client
            .release_key(&h.conn, &qh, &client_kbd, 8, make_key(30));

        host_ptr.motion(2, 50.0, 60.0);
        host_ptr.button(9, 3, 0x111, server_pointer::ButtonState::Pressed);
        host_ptr.button(10, 4, 0x111, server_pointer::ButtonState::Released);
        host_ptr.axis(5, server_pointer::Axis::VerticalScroll, 15.0);
        host_ptr.button(11, 6, 0x110, server_pointer::ButtonState::Released);
        host_ptr.frame();
        h.sync();

        let stalled_events = h.drain_events();
        assert_eq!(
            stalled_events.len(),
            2,
            "only pre-stall key/button releases should forward while stalled: {stalled_events:?}"
        );
        assert!(matches!(
            &stalled_events[0],
            Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                raw_code: 30,
                state: KeyState::Released,
                ..
            }))
        ));
        assert!(matches!(
            &stalled_events[1],
            Event::PointerFrame(frames) if frames.len() == 1 && matches!(
                frames[0].kind,
                PointerEventKind::Release { button: 0x110, .. }
            )
        ));

        // 4. `Request::HeartbeatAck` clears stall -> new key + pointer motion forward again.
        h.client
            .handle_request(RecvType::Object(Request::HeartbeatAck { seq: 1 }));
        h.client
            .press_key(&h.conn, &qh, &client_kbd, 12, make_key(32));
        host_ptr.motion(7, 80.0, 90.0);
        host_ptr.frame();
        h.sync();
        let recovered_events = h.drain_events();
        assert!(recovered_events.iter().any(|e| matches!(
            e,
            Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                raw_code: 32,
                state: KeyState::Pressed,
                ..
            }))
        )));
        assert!(recovered_events.iter().any(|e| matches!(
            e,
            Event::PointerFrame(frames) if frames.iter().any(|p| matches!(
                p.kind,
                PointerEventKind::Motion
            ))
        )));

        // 5. Unacked heartbeat past `STALL_TIMEOUT` returns `TimedOut`, sets
        //    `timed_out()`, and disconnects the serializer cleanly.
        let t_start = Instant::now();
        assert_eq!(
            h.client.handle_heartbeat_tick(t_start),
            HeartbeatTickAction::Continue
        );
        assert_eq!(h.drain_events(), vec![Event::Heartbeat { seq: 2 }]);
        assert!(!h.client.timed_out());

        assert_eq!(
            h.client
                .handle_heartbeat_tick(t_start + constants::STALL_TIMEOUT),
            HeartbeatTickAction::TimedOut
        );
        assert!(h.client.timed_out());
        assert!(!h.client.serializer.other_end_connected());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_scroll_blit_end_to_end_swapchain_damage_and_resync() {
        use std::num::NonZeroUsize;

        use crate::buffer_pointer::BufferPointer;
        use crate::serialization::framing::RawBufferHeader;
        use crate::serialization::framing::ScrollRect;
        use crate::serialization::geometry::Rectangle;
        use crate::serialization::wayland::BufferFormat;
        use crate::serialization::wayland::BufferMetadata;
        use crate::sharding_compression::ShardingCompressor;

        let mut h = ClientDndHarness::new();
        // Force real 2-slot `SlotPool` swapchain ping-pong (compositor holds the
        // currently scanning-out buffer until the next commit replaces it).
        h.host.hold_committed_buffers = true;

        let client_id = ClientId(3);
        let sid = WlSurfaceId(301);
        let width = 640usize;
        let height = 512usize;
        let stride = width * 4;
        let len = height * stride;
        let meta = BufferMetadata {
            width: width as i32,
            height: height as i32,
            stride: stride as i32,
            format: BufferFormat::Xrgb8888,
        };

        let mut doc = vec![0u8; (height + 300) * stride];
        for y in 0..(height + 300) {
            let line = y / 16;
            for x in 0..width {
                let off = y * stride + x * 4;
                let glyph = (3..=12).contains(&(y % 16))
                    && x >= 16
                    && x < width - 16
                    && (((x * 23) ^ (y * 13) ^ (line * 97)) % 5) < 2;
                let px = if glyph {
                    [
                        ((line * 31 + x) & 0x7F) as u8,
                        ((line * 17 + y) & 0x7F) as u8,
                        ((line * 47 + x * 3) & 0x7F) as u8,
                        0xFF,
                    ]
                } else {
                    [248, 248, 250, 0xFF]
                };
                doc[off..off + 4].copy_from_slice(&px);
            }
        }

        let mut compressor = ShardingCompressor::new(NonZeroUsize::new(4).unwrap(), 1).unwrap();
        let mut shadow = Vec::new();
        let mut next_seq = 1u64;

        let send_wire_stream =
            |h: &mut ClientDndHarness,
             mut streaming: crate::serialization::framing::StreamingCompressedTiles,
             seq: u64| {
                streaming.header.frame_seq = seq;
                let mut wire = Vec::new();
                streaming.framed_write(&mut wire).unwrap();
                let (tx, rx) = crossbeam_channel::bounded(1);
                let reader = std::thread::spawn(move || {
                    CompressedTileStream::read_from_stream(&mut wire.as_slice(), |stream| {
                        tx.send(stream).unwrap();
                        Ok(())
                    })
                    .unwrap();
                });
                let stream = rx.recv().unwrap();
                h.client.handle_request(RecvType::RawBuffer(stream));
                reader.join().unwrap();
            };

        let commit_surface = |h: &mut ClientDndHarness, damage: Option<Vec<Rectangle<i32>>>| {
            let state = SurfaceState {
                client: client_id,
                id: sid,
                buffer: Some(BufferAssignment::New(meta)),
                role: None,
                buffer_delta: None,
                buffer_scale: 1,
                buffer_transform: None,
                opaque_region: None,
                input_region: None,
                z_ordered_children: vec![wayland::SubsurfacePosition {
                    id: sid,
                    position: (0, 0).into(),
                }],
                damage,
                output_ids: Vec::new(),
                viewport_state: None,
                xdg_surface_state: None,
            };
            h.client
                .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                    client: client_id,
                    surface: sid,
                    payload: SurfaceRequestPayload::Commit(state),
                })));
            h.sync();
            h.fire_frame_callbacks();
        };

        // 1. Frame 0: initial full sync.
        let mut scroll_y = 100usize;
        let frame0 = doc[scroll_y * stride..scroll_y * stride + len].to_vec();
        {
            let ptr = frame0.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, len) };
            let mut streaming = None;
            filtering::filter_and_compress_full_streaming(
                bp,
                client_id,
                sid,
                meta,
                CompressionMode::Lossless,
                Some(&mut shadow),
                None,
                &mut compressor,
                |s| streaming = Some(s),
            );
            let seq = next_seq;
            next_seq += 1;
            send_wire_stream(&mut h, streaming.unwrap(), seq);
            assert_eq!(h.drain_events(), vec![Event::FrameAck { seq }]);
            commit_surface(&mut h, None);
        }
        let host_surface_id = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&sid)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();

        // 2. 10-frame smooth scroll sequence (`dy = -6, -12, +8, ...`) across the
        //    real 2-slot `SlotPool` swapchain.
        let deltas: [i32; 10] = [-6, -12, -6, -12, 8, 8, -12, 8, -6, -12];
        for (step_idx, expected_dy) in deltas.into_iter().enumerate() {
            h.host.damage_buffers.clear();
            scroll_y = (scroll_y as isize - expected_dy as isize) as usize;
            let frame = doc[scroll_y * stride..scroll_y * stride + len].to_vec();

            let ptr = frame.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, len) };
            let mut streaming = None;
            let diff = filtering::filter_and_compress_incremental_streaming(
                bp,
                client_id,
                sid,
                meta,
                CompressionMode::Lossless,
                None,
                &mut shadow,
                None,
                &mut compressor,
                |s| streaming = Some(s),
            );
            let sr = diff
                .scroll_rect
                .expect("expected ScrollRect on smooth scroll frame");
            assert_eq!(sr.dy, expected_dy, "step {step_idx}");

            let seq = next_seq;
            next_seq += 1;
            send_wire_stream(&mut h, streaming.unwrap(), seq);
            assert_eq!(h.drain_events(), vec![Event::FrameAck { seq }]);

            // On step 5, deliberately omit `sr.dst_damage_rect()` from `Commit.damage`
            // to verify `wprsc`'s `pending_scroll_damage` safety net still damages it!
            let tile_damage: Vec<Rectangle<i32>> = diff
                .dirty_tiles
                .iter()
                .map(|r| Rectangle::new(r.x as i32, r.y as i32, r.w as i32, r.h as i32))
                .collect();
            let commit_damage = if step_idx == 5 {
                tile_damage.clone()
            } else {
                let mut d = Vec::with_capacity(1 + tile_damage.len());
                d.push(sr.dst_damage_rect());
                d.extend_from_slice(&tile_damage);
                d
            };
            commit_surface(&mut h, Some(commit_damage));

            // Verify exact `wl_surface.damage_buffer` rects on the host compositor.
            let got_damage: Vec<Rectangle<i32>> = h
                .host
                .damage_buffers
                .iter()
                .filter(|&&(id, ..)| id == host_surface_id)
                .map(|&(_, x, y, w, hh)| Rectangle::new(x, y, w, hh))
                .collect();
            assert!(
                got_damage.contains(&sr.dst_damage_rect()),
                "step {step_idx}: missing scroll dst damage {:?} in {got_damage:?}",
                sr.dst_damage_rect()
            );
            for tile_rect in &tile_damage {
                assert!(
                    got_damage.contains(tile_rect),
                    "step {step_idx}: missing tile damage {tile_rect:?} in {got_damage:?}"
                );
            }
            assert_eq!(
                got_damage.len(),
                1 + tile_damage.len(),
                "step {step_idx}: duplicate damage rects emitted: {got_damage:?}"
            );

            // Verify bit-exact pixels in `wprsc`'s active `SlotPool` buffer!
            let active_slot = h
                .client
                .remote_display
                .client(&client_id)
                .surfaces
                .get(&sid)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .active_buffer
                .slot();
            let slot_bytes = h.client.pool.raw_data_mut(&active_slot);
            assert_eq!(
                &slot_bytes[..len],
                &frame[..],
                "pixel mismatch in SlotPool on smooth scroll step {step_idx} (dy={expected_dy})"
            );
        }

        // 3. Receiving a `ScrollRect` on a surface with no valid previous buffer
        //    must fail safely and emit `BufferResyncRequested` BEFORE `FrameAck`.
        let unknown_sid = WlSurfaceId(999);
        let (_tx, rx) = crossbeam_channel::unbounded();
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 555,
                    client: client_id,
                    surface: unknown_sid,
                    metadata: meta,
                    incremental: true,
                    scroll_rect: Some(ScrollRect {
                        x: 0,
                        y: 0,
                        width: width as u32,
                        height: height as u32,
                        dy: -16,
                    }),
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![],
                },
                tiles: rx,
            }));
        assert_eq!(
            h.drain_events(),
            vec![
                Event::Surface(SurfaceEvent {
                    surface_id: unknown_sid,
                    payload: SurfaceEventPayload::BufferResyncRequested,
                }),
                Event::FrameAck { seq: 555 },
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_tile_cache_aba_swapchain_damage_resync_lockstep_and_reconnect() {
        use std::num::NonZeroUsize;

        use crate::buffer_pointer::BufferPointer;
        use crate::filtering::ServerTileCache;
        use crate::serialization::framing::CacheHitRef;
        use crate::serialization::framing::CacheHitRegion;
        use crate::serialization::framing::CompressedTile;
        use crate::serialization::framing::LeafPayload;
        use crate::serialization::framing::RawBufferHeader;
        use crate::serialization::framing::SubTile;
        use crate::serialization::framing::TilePayload;
        use crate::serialization::framing::TileRect;
        use crate::serialization::geometry::Rectangle;
        use crate::serialization::wayland::BufferFormat;
        use crate::serialization::wayland::BufferMetadata;
        use crate::sharding_compression::ShardingCompressor;

        let mut h = ClientDndHarness::new();
        // Force real 2-slot `SlotPool` swapchain ping-pong with compositor-held buffers.
        h.host.hold_committed_buffers = true;

        let client_id = ClientId(4);
        let sid_a = WlSurfaceId(401);
        let sid_b = WlSurfaceId(402);
        let width = 512usize;
        let height = 256usize;
        let stride = width * 4;
        let len = height * stride;
        let meta = BufferMetadata {
            width: width as i32,
            height: height as i32,
            stride: stride as i32,
            format: BufferFormat::Xrgb8888,
        };

        let make_frame = |seed: u8| -> Vec<u8> {
            let mut out = vec![0u8; len];
            for y in 0..height {
                for x in 0..width {
                    let off = y * stride + x * 4;
                    let tile_salt = (x >> 8) * 37 + (y >> 8) * 53;
                    out[off] = seed.wrapping_add(((x * 13 + y * 7 + tile_salt) & 0xFF) as u8);
                    out[off + 1] = seed
                        .wrapping_mul(3)
                        .wrapping_add(((x * 5 + y * 11 + tile_salt * 3) & 0xFF) as u8);
                    out[off + 2] = seed
                        .wrapping_mul(7)
                        .wrapping_add(((x * 17 + y * 3 + tile_salt * 5) & 0xFF) as u8);
                    out[off + 3] = 0xFF;
                }
            }
            out
        };
        let frame_a = make_frame(1);
        let frame_b = make_frame(2);

        let mut compressor = ShardingCompressor::new(NonZeroUsize::new(4).unwrap(), 1).unwrap();
        let mut server_cache = ServerTileCache::new();
        let mut shadow_a = Vec::new();

        let send_wire_stream =
            |h: &mut ClientDndHarness,
             mut streaming: crate::serialization::framing::StreamingCompressedTiles,
             seq: u64| {
                streaming.header.frame_seq = seq;
                let mut wire = Vec::new();
                streaming.framed_write(&mut wire).unwrap();
                let (tx, rx) = crossbeam_channel::bounded(1);
                let reader = std::thread::spawn(move || {
                    CompressedTileStream::read_from_stream(&mut wire.as_slice(), |stream| {
                        tx.send(stream).unwrap();
                        Ok(())
                    })
                    .unwrap();
                });
                let stream = rx.recv().unwrap();
                h.client.handle_request(RecvType::RawBuffer(stream));
                reader.join().unwrap();
            };

        let commit_surface =
            |h: &mut ClientDndHarness, sid: WlSurfaceId, damage: Option<Vec<Rectangle<i32>>>| {
                let state = SurfaceState {
                    client: client_id,
                    id: sid,
                    buffer: Some(BufferAssignment::New(meta)),
                    role: None,
                    buffer_delta: None,
                    buffer_scale: 1,
                    buffer_transform: None,
                    opaque_region: None,
                    input_region: None,
                    z_ordered_children: vec![wayland::SubsurfacePosition {
                        id: sid,
                        position: (0, 0).into(),
                    }],
                    damage,
                    output_ids: Vec::new(),
                    viewport_state: None,
                    xdg_surface_state: None,
                };
                h.client
                    .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                        client: client_id,
                        surface: sid,
                        payload: SurfaceRequestPayload::Commit(state),
                    })));
                h.sync();
                h.fire_frame_callbacks();
            };

        // 1. A -> B -> A over real 2-slot `SlotPool` swapchain: return-to-A uses
        //    `CacheHit` tiles, accurately updates `stale_regions` + `damage_buffer`,
        //    and reconstructs bit-exact pixels in `SlotPool`.
        for (step, (frame, full)) in [(&frame_a, true), (&frame_b, false), (&frame_a, false)]
            .into_iter()
            .enumerate()
        {
            h.host.damage_buffers.clear();
            let ptr = frame.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, len) };
            let mut streaming = None;
            let damage = if full {
                filtering::filter_and_compress_full_streaming(
                    bp,
                    client_id,
                    sid_a,
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow_a),
                    Some(&mut server_cache),
                    &mut compressor,
                    |s| streaming = Some(s),
                );
                None
            } else {
                let diff = filtering::filter_and_compress_incremental_streaming(
                    bp,
                    client_id,
                    sid_a,
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow_a,
                    Some(&mut server_cache),
                    &mut compressor,
                    |s| streaming = Some(s),
                );
                Some(
                    diff.dirty_tiles
                        .iter()
                        .map(|r| Rectangle::new(r.x as i32, r.y as i32, r.w as i32, r.h as i32))
                        .collect(),
                )
            };
            let seq = step as u64 + 1;
            send_wire_stream(&mut h, streaming.unwrap(), seq);
            assert_eq!(h.drain_events(), vec![Event::FrameAck { seq }]);
            commit_surface(&mut h, sid_a, damage.clone());

            let host_sid = h
                .client
                .remote_display
                .client(&client_id)
                .surfaces
                .get(&sid_a)
                .unwrap()
                .wl_surface()
                .id()
                .protocol_id();
            if let Some(exp_dmg) = &damage {
                let got_dmg: Vec<Rectangle<i32>> = h
                    .host
                    .damage_buffers
                    .iter()
                    .filter(|&&(id, ..)| id == host_sid)
                    .map(|&(_, x, y, w, hh)| Rectangle::new(x, y, w, hh))
                    .collect();
                for r in exp_dmg {
                    assert!(
                        got_dmg.contains(r),
                        "step {step}: missing CacheHit tile damage {r:?} in {got_dmg:?}"
                    );
                }
            }
            let active_slot = h
                .client
                .remote_display
                .client(&client_id)
                .surfaces
                .get(&sid_a)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .active_buffer
                .slot();
            assert_eq!(
                &h.client.pool.raw_data_mut(&active_slot)[..len],
                frame.as_slice(),
                "SlotPool pixel mismatch on step {step}"
            );
        }
        assert!(h.client.tile_cache.occupied_slots() >= 4);

        // 2. `BufferResyncRequested` lockstep test: an incremental frame for an
        //    unknown surface `sid_missing` triggers `prepare_buffer_canvas` failure,
        //    emits `BufferResyncRequested` BEFORE `FrameAck`, AND still populates
        //    its `CachePut` slot (using a non-zero-origin `Split` tile!) via
        //    `decompress_cache_puts_only` so an already-pipelined frame for `sid_b`
        //    referencing that slot via `CacheHit` succeeds bit-exact!
        let sid_missing = WlSurfaceId(499);
        let put_slot = 500u16;
        let put_rect = TileRect {
            x: 256,
            y: 0,
            w: 256,
            h: 256,
        };
        let split_tile = CompressedTile {
            tile_idx: 0,
            cache_put: Some(put_slot),
            payload: TilePayload::Split(vec![
                SubTile {
                    rect: TileRect {
                        x: 256,
                        y: 0,
                        w: 128,
                        h: 128,
                    },
                    payload: LeafPayload::Solid([0x11, 0x22, 0x33, 0xFF]),
                },
                SubTile {
                    rect: TileRect {
                        x: 384,
                        y: 0,
                        w: 128,
                        h: 128,
                    },
                    payload: LeafPayload::Solid([0x44, 0x55, 0x66, 0xFF]),
                },
                SubTile {
                    rect: TileRect {
                        x: 256,
                        y: 128,
                        w: 128,
                        h: 128,
                    },
                    payload: LeafPayload::Solid([0x77, 0x88, 0x99, 0xFF]),
                },
                SubTile {
                    rect: TileRect {
                        x: 384,
                        y: 128,
                        w: 128,
                        h: 128,
                    },
                    payload: LeafPayload::Solid([0xAA, 0xBB, 0xCC, 0xFF]),
                },
            ]),
        };
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(Ok(split_tile)).unwrap();
        drop(tx);
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 901,
                    client: client_id,
                    surface: sid_missing,
                    metadata: meta,
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![put_rect],
                },
                tiles: rx,
            }));
        assert_eq!(
            h.drain_events(),
            vec![
                Event::Surface(SurfaceEvent {
                    surface_id: sid_missing,
                    payload: SurfaceEventPayload::BufferResyncRequested,
                }),
                Event::FrameAck { seq: 901 },
            ]
        );

        // Now pipelined surface `sid_b` arrives as a full frame referencing `put_slot`
        // for both `(0,0,256,256)` and `(256,0,256,256)` via `CacheHit` -> MUST succeed!
        let (tx, rx) = crossbeam_channel::unbounded();
        for tile_idx in 0..2u32 {
            tx.send(Ok(CompressedTile {
                tile_idx,
                cache_put: None,
                payload: TilePayload::CacheHit(CacheHitRef {
                    slot_id: put_slot,
                    region: CacheHitRegion::Full,
                }),
            }))
            .unwrap();
        }
        drop(tx);
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 902,
                    client: client_id,
                    surface: sid_b,
                    metadata: meta,
                    incremental: false,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![
                        TileRect {
                            x: 0,
                            y: 0,
                            w: 256,
                            h: 256,
                        },
                        put_rect,
                    ],
                },
                tiles: rx,
            }));
        assert_eq!(h.drain_events(), vec![Event::FrameAck { seq: 902 }]);
        let b_slot = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&sid_b)
            .unwrap()
            .buffer
            .as_ref()
            .unwrap()
            .active_buffer
            .slot();
        let b_bytes = h.client.pool.raw_data_mut(&b_slot);
        // Spot-check all 4 quadrants of the Split tile decoded via `puts_only` and blitted via `CacheHit`.
        assert_eq!(&b_bytes[0..4], &[0x11, 0x22, 0x33, 0xFF]);
        assert_eq!(&b_bytes[128 * 4..128 * 4 + 4], &[0x44, 0x55, 0x66, 0xFF]);
        assert_eq!(
            &b_bytes[128 * stride..128 * stride + 4],
            &[0x77, 0x88, 0x99, 0xFF]
        );
        assert_eq!(
            &b_bytes[128 * stride + 128 * 4..128 * stride + 128 * 4 + 4],
            &[0xAA, 0xBB, 0xCC, 0xFF]
        );

        // 3. Missing `CacheHit` slot triggers clean `BufferResyncRequested` (never panics).
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(Ok(CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::CacheHit(CacheHitRef {
                slot_id: 999,
                region: CacheHitRegion::Full,
            }),
        }))
        .unwrap();
        drop(tx);
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 903,
                    client: client_id,
                    surface: sid_a,
                    metadata: meta,
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 256,
                        h: 256,
                    }],
                },
                tiles: rx,
            }));
        assert_eq!(
            h.drain_events(),
            vec![
                Event::Surface(SurfaceEvent {
                    surface_id: sid_a,
                    payload: SurfaceEventPayload::BufferResyncRequested,
                }),
                Event::FrameAck { seq: 903 },
            ]
        );

        // 4. `header.reset_tile_cache` clears `wprsc.tile_cache` in-band, and
        //    `Request::Capabilities` clears `wprsc.tile_cache` on reconnect.
        assert!(h.client.tile_cache.occupied_slots() > 0);
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(Ok(CompressedTile {
            tile_idx: 0,
            cache_put: Some(0),
            payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
        }))
        .unwrap();
        drop(tx);
        h.client
            .handle_request(RecvType::RawBuffer(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 904,
                    client: client_id,
                    surface: sid_b,
                    metadata: meta,
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: true,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 64,
                        h: 64,
                    }],
                },
                tiles: rx,
            }));
        assert_eq!(h.drain_events(), vec![Event::FrameAck { seq: 904 }]);
        assert_eq!(h.client.tile_cache.occupied_slots(), 1);

        h.client
            .handle_request(RecvType::Object(Request::Capabilities(Capabilities {
                xwayland: false,
            })));
        assert_eq!(h.client.tile_cache.occupied_slots(), 0);
        assert_eq!(h.client.tile_cache.used_bytes(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsc_cropped_subtile_scroll_scrollbar_and_odd_edits_swapchain() {
        use std::num::NonZeroUsize;

        use crate::buffer_pointer::BufferPointer;
        use crate::filtering;
        use crate::filtering::ServerTileCache;
        use crate::serialization::framing::CompressedTileStream;
        use crate::serialization::framing::CompressionMode;
        use crate::serialization::framing::TileRect;
        use crate::serialization::geometry::Rectangle;
        use crate::serialization::wayland::BufferFormat;
        use crate::serialization::wayland::BufferMetadata;
        use crate::sharding_compression::ShardingCompressor;

        let mut h = ClientDndHarness::new();
        h.host.hold_committed_buffers = true;

        let client_id = ClientId(8);
        let sid = WlSurfaceId(801);
        let width = 512usize;
        let height = 512usize;
        let stride = width * 4;
        let len = height * stride;
        let sb_w = 12usize;
        let meta = BufferMetadata {
            width: width as i32,
            height: height as i32,
            stride: stride as i32,
            format: BufferFormat::Xrgb8888,
        };

        let mut doc = vec![0u8; (height + 160) * stride];
        for y in 0..(height + 160) {
            for x in 0..width {
                let off = y * stride + x * 4;
                let line = y / 16;
                let glyph = (3..=12).contains(&(y % 16))
                    && x >= 12
                    && x < width - 20
                    && (((x * 19) ^ (y * 11) ^ (line * 71)) % 5) < 2;
                let px = if glyph {
                    [
                        ((line * 29 + x) & 0x7F) as u8,
                        ((line * 13 + y) & 0x7F) as u8,
                        ((line * 43 + x * 3) & 0x7F) as u8,
                        0xFF,
                    ]
                } else {
                    [246, 248, 250, 0xFF]
                };
                doc[off..off + 4].copy_from_slice(&px);
            }
        }
        let make_frame = |scroll_y: usize, thumb_y: usize| -> Vec<u8> {
            let mut f = doc[scroll_y * stride..scroll_y * stride + len].to_vec();
            for y in 0..height {
                let in_thumb = y >= thumb_y && y < thumb_y + 84;
                for x in (width - sb_w)..width {
                    let off = y * stride + x * 4;
                    f[off..off + 4].copy_from_slice(if in_thumb {
                        &[115, 118, 125, 0xFF]
                    } else {
                        &[228, 230, 234, 0xFF]
                    });
                }
            }
            f
        };

        let mut compressor = ShardingCompressor::new(NonZeroUsize::new(4).unwrap(), 1).unwrap();
        let mut srv_cache = ServerTileCache::new();
        let mut shadow = Vec::new();
        let mut next_seq = 1u64;

        let send_wire_stream =
            |h: &mut ClientDndHarness,
             mut streaming: crate::serialization::framing::StreamingCompressedTiles,
             seq: u64| {
                streaming.header.frame_seq = seq;
                let mut wire = Vec::new();
                streaming.framed_write(&mut wire).unwrap();
                let (tx, rx) = crossbeam_channel::bounded(1);
                let reader = std::thread::spawn(move || {
                    CompressedTileStream::read_from_stream(&mut wire.as_slice(), |stream| {
                        tx.send(stream).unwrap();
                        Ok(())
                    })
                    .unwrap();
                });
                let stream = rx.recv().unwrap();
                h.client.handle_request(RecvType::RawBuffer(stream));
                reader.join().unwrap();
            };

        let commit_surface = |h: &mut ClientDndHarness, damage: Option<Vec<Rectangle<i32>>>| {
            let state = SurfaceState {
                client: client_id,
                id: sid,
                buffer: Some(BufferAssignment::New(meta)),
                role: None,
                buffer_delta: None,
                buffer_scale: 1,
                buffer_transform: None,
                opaque_region: None,
                input_region: None,
                z_ordered_children: vec![wayland::SubsurfacePosition {
                    id: sid,
                    position: (0, 0).into(),
                }],
                damage,
                output_ids: Vec::new(),
                viewport_state: None,
                xdg_surface_state: None,
            };
            h.client
                .handle_request(RecvType::Object(Request::Surface(SurfaceRequest {
                    client: client_id,
                    surface: sid,
                    payload: SurfaceRequestPayload::Commit(state),
                })));
            h.sync();
            h.fire_frame_callbacks();
        };

        let assert_active_slot_eq = |h: &mut ClientDndHarness, expected: &[u8], label: &str| {
            let slot = h
                .client
                .remote_display
                .client(&client_id)
                .surfaces
                .get(&sid)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .active_buffer
                .slot();
            let bytes = h.client.pool.raw_data_mut(&slot);
            assert_eq!(&bytes[..len], expected, "SlotPool mismatch on {label}");
        };

        // Step 0: Initial full sync seeds both swapchain and tile cache.
        let f0 = make_frame(40, 32);
        {
            let p = f0.as_ptr();
            let bp = unsafe { BufferPointer::new(&p, len) };
            let mut st = None;
            filtering::filter_and_compress_full_streaming(
                bp,
                client_id,
                sid,
                meta,
                CompressionMode::Lossless,
                Some(&mut shadow),
                Some(&mut srv_cache),
                &mut compressor,
                |s| st = Some(s),
            );
            let seq = next_seq;
            next_seq += 1;
            send_wire_stream(&mut h, st.unwrap(), seq);
            assert_eq!(h.drain_events(), vec![Event::FrameAck { seq }]);
            commit_surface(&mut h, None);
            assert_active_slot_eq(&mut h, &f0, "initial full frame");
        }
        let host_surface_id = h
            .client
            .remote_display
            .client(&client_id)
            .surfaces
            .get(&sid)
            .unwrap()
            .wl_surface()
            .id()
            .protocol_id();
        let initial_cache_slots = h.client.tile_cache.occupied_slots();

        // Steps 1..=3: Smooth scrolls (`dy = -12, -12, +12`) with moving 12px
        // overlay scrollbar across 2-slot `SlotPool` ping-pong.
        let scroll_steps = [(52usize, 44usize, -12i32), (64, 56, -12), (52, 44, 12)];
        let mut last_frame = f0;
        for (idx, (sy, ty, expected_dy)) in scroll_steps.into_iter().enumerate() {
            h.host.damage_buffers.clear();
            let frame = make_frame(sy, ty);
            let p = frame.as_ptr();
            let bp = unsafe { BufferPointer::new(&p, len) };
            let mut st = None;
            let diff = filtering::filter_and_compress_incremental_streaming(
                bp,
                client_id,
                sid,
                meta,
                CompressionMode::Lossless,
                None,
                &mut shadow,
                Some(&mut srv_cache),
                &mut compressor,
                |s| st = Some(s),
            );
            let sr = diff.scroll_rect.expect("expected ScrollRect");
            assert_eq!(sr.dy, expected_dy, "scroll step {idx}");
            assert!(
                diff.dirty_tiles
                    .iter()
                    .any(|r| r.x == (width - sb_w) as u32 && r.w == sb_w as u32),
                "step {idx}: expected tight 12px scrollbar rect in {:?}",
                diff.dirty_tiles
            );
            let seq = next_seq;
            next_seq += 1;
            send_wire_stream(&mut h, st.unwrap(), seq);
            assert_eq!(h.drain_events(), vec![Event::FrameAck { seq }]);
            let mut damage = vec![sr.dst_damage_rect()];
            damage.extend(
                diff.dirty_tiles
                    .iter()
                    .map(|r| Rectangle::new(r.x as i32, r.y as i32, r.w as i32, r.h as i32)),
            );
            commit_surface(&mut h, Some(damage.clone()));

            let got_damage: Vec<Rectangle<i32>> = h
                .host
                .damage_buffers
                .iter()
                .filter(|&&(id, ..)| id == host_surface_id)
                .map(|&(_, x, y, w, hh)| Rectangle::new(x, y, w, hh))
                .collect();
            for rect in &damage {
                assert!(
                    got_damage.contains(rect),
                    "step {idx}: missing damage_buffer {rect:?} in {got_damage:?}"
                );
            }
            assert_eq!(h.client.tile_cache.occupied_slots(), initial_cache_slots);
            assert_active_slot_eq(&mut h, &frame, &format!("scroll step {idx}"));
            last_frame = frame;
        }

        // Steps 4..=5: Odd-aligned `1x1` and `13x17` cropped edits on alternating
        // swapchain slots verify `stale_regions` synchronization with arbitrary
        // odd-sized `TileRect`s.
        let mut edit1 = last_frame;
        edit1[73 * stride + 91 * 4] ^= 0x5A;
        let mut edit2 = edit1.clone();
        for y in 201..218 {
            for x in 157..170 {
                edit2[y * stride + x * 4 + 1] ^= 0x3C;
            }
        }
        for (label, frame, expected_rect) in [
            (
                "odd 1x1",
                &edit1,
                TileRect {
                    x: 91,
                    y: 73,
                    w: 1,
                    h: 1,
                },
            ),
            (
                "odd 13x17",
                &edit2,
                TileRect {
                    x: 157,
                    y: 201,
                    w: 13,
                    h: 17,
                },
            ),
        ] {
            h.host.damage_buffers.clear();
            let p = frame.as_ptr();
            let bp = unsafe { BufferPointer::new(&p, len) };
            let mut st = None;
            let diff = filtering::filter_and_compress_incremental_streaming(
                bp,
                client_id,
                sid,
                meta,
                CompressionMode::Lossless,
                None,
                &mut shadow,
                Some(&mut srv_cache),
                &mut compressor,
                |s| st = Some(s),
            );
            assert_eq!(diff.scroll_rect, None);
            assert_eq!(diff.dirty_tiles, vec![expected_rect], "{label}");
            let seq = next_seq;
            next_seq += 1;
            send_wire_stream(&mut h, st.unwrap(), seq);
            assert_eq!(h.drain_events(), vec![Event::FrameAck { seq }]);
            let dmg_rect = Rectangle::new(
                expected_rect.x as i32,
                expected_rect.y as i32,
                expected_rect.w as i32,
                expected_rect.h as i32,
            );
            commit_surface(&mut h, Some(vec![dmg_rect]));
            let got_damage: Vec<Rectangle<i32>> = h
                .host
                .damage_buffers
                .iter()
                .filter(|&&(id, ..)| id == host_surface_id)
                .map(|&(_, x, y, w, hh)| Rectangle::new(x, y, w, hh))
                .collect();
            assert_eq!(got_damage, vec![dmg_rect], "{label} damage_buffer");
            assert_eq!(h.client.tile_cache.occupied_slots(), initial_cache_slots);
            assert_active_slot_eq(&mut h, frame, label);
        }
    }
}
