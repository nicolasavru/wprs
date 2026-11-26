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

            remote_surface.frame_callback_completed
        };

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
                &self.qh,
                &self.title_prefix,
            )
            .location(loc!())?,
            Some(wayland::Role::XdgPopup(_)) => RemoteXdgPopup::apply(
                surface_state,
                surface_id,
                surfaces,
                &self.xdg_shell_state,
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
                ToplevelRequestPayload::SetFullscreen => {
                    toplevel.local_window.set_fullscreen(None);
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
        self.capabilities.set(caps).location(loc!())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_buffer(&mut self, buffer_stream: CompressedTileStream) -> Result<()> {
        let client_id = buffer_stream.header.client;
        let surface_id = buffer_stream.header.surface;

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

            let canvas = remote_surface
                .prepare_buffer_canvas(
                    metadata,
                    validated.incremental_tile_rects(),
                    &mut self.pool,
                    &mut self.slot_copy_scratch,
                )
                .location(loc!())?;
            filtering::decompress_and_unfilter_streaming(validated, &mut self.decompressor, canvas)
                .location(loc!())
        })();

        if let Err(err) = decode_res {
            self.request_buffer_resync(client_id, surface_id);
            return Err(err);
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    pub fn handle_request(&mut self, request: RecvType<Request>) {
        match request {
            RecvType::Connected => unreachable!("client never receives RecvType::Connected"),
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
            RecvType::RawBuffer(buffer) => self.handle_buffer(buffer),
        }
        .log_and_ignore(loc!())
        // TODO: maybe send errors back to the server.
    }
}

#[cfg(test)]
mod tests {
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
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_surface::XdgSurface as ServerXdgSurface;
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
    use smithay::reexports::wayland_server::protocol::wl_subsurface::WlSubsurface as ServerWlSubsurface;
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

    #[derive(Default)]
    struct MockHostState {
        client: Option<ServerClient>,
        pointer: Option<ServerWlPointer>,
        data_device: Option<ServerWlDataDevice>,
        surfaces: Vec<ServerWlSurface>,
        last_source: Option<ServerWlDataSource>,
        receive_fds: Vec<(String, OwnedFd)>,
        reqs: Vec<HostRecordedReq>,
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
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlSurface,
            _request: <ServerWlSurface as Resource>::Request,
            _data: &(),
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
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlSubcompositor,
            request: wl_subcompositor::Request,
            _data: &(),
            _dh: &DisplayHandle,
            data_init: &mut DataInit<'_, Self>,
        ) {
            if let wl_subcompositor::Request::GetSubsurface { id, .. } = request {
                data_init.init(id, ());
            }
        }
    }

    impl ServerDispatch<ServerWlSubsurface, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerWlSubsurface,
            _request: <ServerWlSubsurface as Resource>::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
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
            if let xdg_wm_base::Request::GetXdgSurface { id, .. } = request {
                data_init.init(id, ());
            }
        }
    }

    impl ServerDispatch<ServerXdgSurface, ()> for MockHostState {
        fn request(
            _state: &mut Self,
            _client: &ServerClient,
            _resource: &ServerXdgSurface,
            _request: <ServerXdgSurface as Resource>::Request,
            _data: &(),
            _dh: &DisplayHandle,
            _data_init: &mut DataInit<'_, Self>,
        ) {
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
            seat.capabilities(server_seat::Capability::Pointer);
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
            if let server_seat::Request::GetPointer { id } = request {
                state.pointer = Some(data_init.init(id, ()));
            }
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
            let mut display: Display<MockHostState> = Display::new().unwrap();
            let mut dh = display.handle();
            dh.create_global::<MockHostState, ServerWlShm, _>(1, ());
            dh.create_global::<MockHostState, ServerWlCompositor, _>(4, ());
            dh.create_global::<MockHostState, ServerWlSubcompositor, _>(1, ());
            dh.create_global::<MockHostState, ServerXdgWmBase, _>(2, ());
            dh.create_global::<MockHostState, ServerWlDataDeviceManager, _>(3, ());
            dh.create_global::<MockHostState, ServerWlSeat, _>(7, ());

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
            // Pump seat capabilities -> pointer + data_device creation.
            harness.sync();
            harness.sync();
            harness.drain_events();
            harness.host.reqs.clear();
            harness
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
}
