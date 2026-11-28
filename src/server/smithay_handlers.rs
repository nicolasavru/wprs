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
use std::time::Instant;

use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::input::Seat;
use smithay::input::SeatHandler;
use smithay::input::SeatState;
use smithay::input::pointer::CursorImageStatus as SmithayCursorImageStatus;
use smithay::output::Output;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as XdgDecorationMode;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::Mode as KdeDecorationMode;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::OrgKdeKwinServerDecoration;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::WEnum;
use smithay::reexports::wayland_server::backend::ClientData;
use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::backend::DisconnectReason;
use smithay::reexports::wayland_server::protocol::wl_buffer;
use smithay::reexports::wayland_server::protocol::wl_output;
use smithay::reexports::wayland_server::protocol::wl_seat;
use smithay::reexports::wayland_server::protocol::wl_surface;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Logical;
use smithay::utils::Point;
use smithay::utils::Serial;
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor;
use smithay::wayland::compositor::BufferAssignment as SmithayBufferAssignment;
use smithay::wayland::compositor::CompositorClientState;
use smithay::wayland::compositor::CompositorHandler;
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::compositor::Damage;
use smithay::wayland::compositor::SubsurfaceCachedState;
use smithay::wayland::compositor::SurfaceAttributes;
use smithay::wayland::compositor::SurfaceData;
use smithay::wayland::compositor::TraversalAction;
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
use crate::serialization::xdg_shell::FullscreenState;
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
use crate::server::FrameBackpressure;
use crate::server::LockedSurfaceState;
use crate::server::PendingDamage;
use crate::server::SyncedSurfaceBuffer;
use crate::server::WprsServerState;

pub(crate) enum ScanDamage<'a> {
    Full,
    Raw(&'a [Damage]),
    BufferRects(&'a [Rectangle<i32>]),
}

pub(crate) enum EncodeBufferOutcome {
    Sent {
        metadata: BufferMetadata,
        damage: Option<Vec<Rectangle<i32>>>,
    },
    Unchanged,
    FaultedAfterSend,
    Failed(anyhow::Error),
}

impl WprsServerState {
    pub(crate) fn encode_and_send_buffer(
        &mut self,
        wl_buffer: &wl_buffer::WlBuffer,
        surface_state: &mut SurfaceState,
        scan_damage: ScanDamage<'_>,
    ) -> EncodeBufferOutcome {
        enum InnerOutcome {
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
        let prev_synced = self.synced_buffers.remove(&surface_state.id);
        let mut raw_buffer_sent = false;
        let writer = self.serializer.writer();
        let frame_backpressure = &mut self.frame_backpressure;
        let compressor = &mut self.compressor;
        let tile_cache = &mut self.tile_cache;
        let compression_mode = self.compression_mode;

        let commit_buf_res =
            compositor_utils::with_buffer_contents(wl_buffer, |data, spec| -> Result<_> {
                let metadata = surface_state.set_buffer(&spec).location(loc!())?;
                match prev_synced {
                    Some(mut synced) if synced.metadata == metadata => {
                        // Use partial row-bands when `< 75%` of height is
                        // damaged to avoid scanning undamaged rows, while
                        // still running full-frame incremental diffing on
                        // `>= 75%` or undamaged commits to skip unchanged
                        // `128x128` quadrants.
                        let scan_bands = match scan_damage {
                            ScanDamage::Full
                            | ScanDamage::Raw([])
                            | ScanDamage::BufferRects([]) => None,
                            ScanDamage::Raw(raw) => {
                                let damage = compositor_utils::damage_to_buffer_rects(
                                    raw,
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
                            },
                            ScanDamage::BufferRects(rects) => filtering::compute_partial_row_bands(
                                rects,
                                metadata.width,
                                metadata.height,
                            ),
                        };
                        compositor_utils::fault_probe_buffer(
                            data,
                            metadata.stride as usize,
                            scan_bands.as_deref(),
                        );
                        let diff = filtering::filter_and_compress_incremental_streaming(
                            data,
                            surface_state.client,
                            surface_state.id,
                            metadata,
                            compression_mode,
                            scan_bands.as_deref(),
                            &mut synced.shadow,
                            Some(tile_cache),
                            compressor,
                            |mut raw_buffer_to_send| {
                                raw_buffer_to_send.header.frame_seq =
                                    frame_backpressure.allocate_seq();
                                raw_buffer_sent = true;
                                writer.send(SendType::RawBuffer(raw_buffer_to_send));
                            },
                        );
                        if !synced.shadow.is_compressed() {
                            synced.last_active = Instant::now();
                        }
                        if diff.is_unchanged() {
                            return Ok(InnerOutcome::Unchanged { synced });
                        }
                        let total_damage =
                            usize::from(diff.scroll_rect.is_some()) + diff.dirty_tiles.len();
                        let damage = (total_damage < constants::SENT_DAMAGE_LIMIT).then(|| {
                            let mut rects = Vec::with_capacity(total_damage);
                            if let Some(sr) = diff.scroll_rect {
                                rects.push(sr.dst_damage_rect());
                            }
                            rects.extend(diff.dirty_tiles.into_iter().map(|r| {
                                Rectangle::new(r.x as i32, r.y as i32, r.w as i32, r.h as i32)
                            }));
                            rects
                        });
                        Ok(InnerOutcome::Sent {
                            synced,
                            metadata,
                            damage,
                        })
                    },
                    other => {
                        let mut shadow =
                            other.map_or_else(filtering::ShadowBuffer::new, |old| old.shadow);
                        compositor_utils::fault_probe_buffer(data, metadata.stride as usize, None);
                        filtering::filter_and_compress_full_streaming(
                            data,
                            surface_state.client,
                            surface_state.id,
                            metadata,
                            compression_mode,
                            Some(&mut shadow),
                            Some(tile_cache),
                            compressor,
                            |mut raw_buffer_to_send| {
                                raw_buffer_to_send.header.frame_seq =
                                    frame_backpressure.allocate_seq();
                                raw_buffer_sent = true;
                                writer.send(SendType::RawBuffer(raw_buffer_to_send));
                            },
                        );
                        Ok(InnerOutcome::Sent {
                            synced: SyncedSurfaceBuffer {
                                metadata,
                                shadow,
                                last_active: Instant::now(),
                            },
                            metadata,
                            damage: None,
                        })
                    },
                }
            });

        match commit_buf_res {
            Ok(Ok(InnerOutcome::Sent {
                synced,
                metadata,
                damage,
            })) => {
                self.synced_buffers.insert(surface_state.id, synced);
                EncodeBufferOutcome::Sent { metadata, damage }
            },
            Ok(Ok(InnerOutcome::Unchanged { synced })) => {
                self.synced_buffers.insert(surface_state.id, synced);
                EncodeBufferOutcome::Unchanged
            },
            Ok(Err(err)) => EncodeBufferOutcome::Failed(err),
            Err(err) => {
                // Smithay reports SIGBUS as `Err(BadMap)` after the closure
                // returns, so `raw_buffer_sent` may already be `true`.
                if raw_buffer_sent {
                    warn!(
                        "shm buffer faulted during commit ({err:?}); detaching remote surface buffer"
                    );
                    surface_state.buffer = None;
                    self.tile_cache.reset_for_peer();
                    EncodeBufferOutcome::FaultedAfterSend
                } else {
                    EncodeBufferOutcome::Failed(anyhow!(err))
                }
            },
        }
    }
}

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

    pub(crate) fn output_id_for_wl_output(&self, wl_output: &wl_output::WlOutput) -> Option<u32> {
        let output = Output::from_resource(wl_output)?;
        self.outputs
            .iter()
            .find_map(|(&id, (candidate, _))| (*candidate == output).then_some(id))
    }

    pub(crate) fn client_wl_output_for_id(
        &self,
        surface: &WlSurface,
        output_id: u32,
    ) -> Option<wl_output::WlOutput> {
        let client = surface.client()?;
        self.outputs
            .get(&output_id)
            .and_then(|(output, _)| output.client_outputs(&client).next())
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
        let fullscreen_output = match fullscreen {
            Some(FullscreenState::Set(output_id)) => {
                output_id.and_then(|id| self.client_wl_output_for_id(toplevel.wl_surface(), id))
            },
            _ => None,
        };
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
                Some(FullscreenState::Set(_)) => {
                    state.states.set(xdg_toplevel::State::Fullscreen);
                },
                Some(FullscreenState::Unset) => {
                    state.states.unset(xdg_toplevel::State::Fullscreen);
                },
                None => {},
            }
            state.fullscreen_output = fullscreen_output;
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
            self.frame_backpressure.remove_deferred(surface_id);
            self.deferred_sync_ancestors.remove(&surface_id);
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
                tracked.pending_buffer_delta = None;
                tracked.pending_damage = PendingDamage::None;
            });

            self.send_toplevel_request(&toplevel, ToplevelRequestPayload::Destroyed);
            self.committed_surfaces.remove(&surface_id);
            self.release_held_frame_callbacks(surface_id);
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
            self.frame_backpressure.remove_deferred(surface_id);
            self.deferred_sync_ancestors.remove(&surface_id);
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
                tracked.pending_buffer_delta = None;
                tracked.pending_damage = PendingDamage::None;
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
            self.release_held_frame_callbacks(surface_id);
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
        output: Option<wl_output::WlOutput>,
    ) {
        let output_id = output
            .as_ref()
            .and_then(|wl_output| self.output_id_for_wl_output(wl_output));
        self.update_state_and_send_toplevel_request(
            &surface,
            |toplevel_state| toplevel_state.fullscreen = Some(FullscreenState::Set(output_id)),
            ToplevelRequestPayload::SetFullscreen(output_id),
        );
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        self.update_state_and_send_toplevel_request(
            &surface,
            |toplevel_state| toplevel_state.fullscreen = Some(FullscreenState::Unset),
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
                if tracked.state.buffer.is_some()
                    && !self.frame_backpressure.is_deferred(surface_id)
                {
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

    fn show_window_menu(
        &mut self,
        surface: ToplevelSurface,
        _seat: wl_seat::WlSeat,
        serial: Serial,
        location: Point<i32, Logical>,
    ) {
        let Some(client_serial) = self.serial_map.get(serial) else {
            warn!("Received show_window_menu request with unknown serial {serial:?}.");
            return;
        };

        self.send_toplevel_request(
            &surface,
            ToplevelRequestPayload::ShowWindowMenu {
                serial: client_serial,
                location: location.into(),
            },
        );
    }
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

pub(crate) fn take_dirty_sync_descendants<N>(
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

pub(crate) fn has_deferred_sync_descendant(
    surface: &WlSurface,
    frame_backpressure: &FrameBackpressure,
) -> bool {
    for child in compositor::get_children(surface) {
        if !compositor::is_sync_subsurface(&child) {
            continue;
        }
        if frame_backpressure.is_deferred(WlSurfaceId::new(&child))
            || has_deferred_sync_descendant(&child, frame_backpressure)
        {
            return true;
        }
    }
    false
}

#[instrument(ret, level = "debug")]
pub(crate) fn get_surface_z_order(surface: &WlSurface) -> Vec<SubsurfacePosition> {
    let mut surface_order = Vec::new();
    // `with_surface_tree_upward` iterates `PrivateSurfaceData::children`
    // bottom-to-top (deepest to topmost), including `surface` itself in its
    // true stacking position relative to its direct children. Returning
    // `SkipChildren` at depth 1 skips grandchildren while still running the
    // processor on each direct child.
    compositor::with_surface_tree_upward(
        surface,
        true,
        |_s, _data, &is_root| {
            if is_root {
                TraversalAction::DoChildren(false)
            } else {
                TraversalAction::SkipChildren
            }
        },
        |s, surface_data, &is_root| {
            let position = if is_root {
                (0, 0).into()
            } else {
                surface_data
                    .cached_state
                    .get::<SubsurfaceCachedState>()
                    .pending()
                    .location
                    .into()
            };
            surface_order.push(SubsurfacePosition {
                id: WlSurfaceId::new(s),
                position,
            });
        },
        |_, _, _| true,
    );
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

    let was_deferred_before = state.frame_backpressure.is_deferred(surface_id);
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
        if state.serializer.other_end_connected()
            && has_deferred_sync_descendant(surface, &state.frame_backpressure)
        {
            state.deferred_sync_ancestors.insert(surface_id);
        } else {
            state.deferred_sync_ancestors.remove(&surface_id);
        }
    }
    on_commit_buffer_handler::<WprsServerState>(surface);
    if was_deferred_before && !state.frame_backpressure.is_deferred(surface_id) {
        state.flush_ready_deferred_sync_ancestors();
    }
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

    sync_subsurface_role(
        &mut tracked.state,
        matches!(surface_data.role, Some("subsurface")),
        parent.as_ref().map(WlSurfaceId::new),
    );

    tracked.state.z_ordered_children = surface_order;

    let mut guard = surface_data.cached_state.get::<ViewportCachedState>();
    let viewport_state = guard.current();

    let mut guard = surface_data.cached_state.get::<SurfaceAttributes>();
    let surface_attributes = guard.current();
    compositor_utils::cap_surface_damage(&mut surface_attributes.damage);
    let frame_callbacks = mem::take(&mut surface_attributes.frame_callbacks);
    let frame_ready_at = state.frame_callback_ready_at(Instant::now());
    let buffer_delta = surface_attributes
        .buffer_delta
        .take()
        .filter(|delta| delta.x != 0 || delta.y != 0)
        .map(Into::into);
    if connected {
        tracked.add_buffer_delta(buffer_delta);
    } else {
        tracked.pending_buffer_delta = None;
        tracked.pending_damage = PendingDamage::None;
    }

    set_regions(surface_attributes, &mut tracked.state);
    set_transformation(surface_attributes, &mut tracked.state);
    set_viewport_state(viewport_state, &mut tracked.state);
    set_xdg_surface_attributes(surface_data, &mut tracked.state);

    match &mut tracked.state.role {
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
            if !connected {
                let res = compositor_utils::with_buffer_contents(buffer, |_, spec| {
                    tracked.state.set_buffer(&spec)
                });
                state.schedule_or_fire_frame_callbacks(
                    surface.clone(),
                    surface_id,
                    frame_callbacks,
                    frame_ready_at,
                );
                match res {
                    Ok(Ok(_)) => {
                        surface_attributes.damage.clear();
                    },
                    Ok(Err(err)) => return Err(err).location(loc!()),
                    Err(err) => return Err(err).location(loc!()),
                }
            } else if !state.frame_backpressure.has_credit()
                || state.frame_backpressure.is_deferred(surface_id)
            {
                let prev_meta = match tracked.state.buffer {
                    Some(BufferAssignment::New(meta)) => Some(meta),
                    _ => None,
                };
                let meta_res = compositor_utils::with_buffer_contents(buffer, |_, spec| {
                    tracked.state.set_buffer(&spec)
                });
                let metadata = match meta_res {
                    Ok(Ok(metadata)) => metadata,
                    Ok(Err(err)) => {
                        state.schedule_or_fire_frame_callbacks(
                            surface.clone(),
                            surface_id,
                            frame_callbacks,
                            frame_ready_at,
                        );
                        return Err(err).location(loc!());
                    },
                    Err(err) => {
                        state.schedule_or_fire_frame_callbacks(
                            surface.clone(),
                            surface_id,
                            frame_callbacks,
                            frame_ready_at,
                        );
                        return Err(err).location(loc!());
                    },
                };
                if prev_meta != Some(metadata) || surface_attributes.damage.is_empty() {
                    tracked.pending_damage.mark_full();
                } else {
                    let rects = compositor_utils::damage_to_buffer_rects(
                        &surface_attributes.damage,
                        tracked.state.buffer_scale,
                        tracked
                            .state
                            .buffer_transform
                            .unwrap_or(Transform::Normal)
                            .into(),
                        tracked.state.viewport_state.as_ref(),
                        (metadata.width, metadata.height),
                    );
                    tracked.pending_damage.merge(rects);
                }
                surface_attributes.damage.clear();
                tracked.has_committed = true;
                tracked.dirty = true;
                state.frame_backpressure.defer(surface_id);
                state.hold_frame_callbacks(surface, surface_id, frame_callbacks, frame_ready_at);
                return Ok(false);
            } else {
                state.schedule_or_fire_frame_callbacks(
                    surface.clone(),
                    surface_id,
                    frame_callbacks,
                    frame_ready_at,
                );
                let outcome = state.encode_and_send_buffer(
                    buffer,
                    &mut tracked.state,
                    ScanDamage::Raw(&surface_attributes.damage),
                );
                match outcome {
                    EncodeBufferOutcome::Sent { metadata, damage } => {
                        surface_attributes.damage.clear();
                        damage_to_send = damage;
                        buffer_to_send = Some(BufferAssignment::New(metadata));
                    },
                    EncodeBufferOutcome::Unchanged => {
                        surface_attributes.damage.clear();
                    },
                    EncodeBufferOutcome::FaultedAfterSend => {
                        surface_attributes.damage.clear();
                        buffer_to_send = Some(BufferAssignment::Removed);
                    },
                    EncodeBufferOutcome::Failed(err) => {
                        return Err(err).location(loc!());
                    },
                }
            }
        },
        Some(SmithayBufferAssignment::Removed) => {
            surface_attributes.damage.clear();
            tracked.pending_damage = PendingDamage::None;
            tracked.pending_buffer_delta = None;
            tracked.state.buffer = None;
            state.synced_buffers.remove(&surface_id);
            if state.frame_backpressure.remove_deferred(surface_id) {
                state.release_held_frame_callbacks(surface_id);
            }
            state.schedule_or_fire_frame_callbacks(
                surface.clone(),
                surface_id,
                frame_callbacks,
                frame_ready_at,
            );
            buffer_to_send = Some(BufferAssignment::Removed);
        },
        // Retain `surface_attributes.damage` across bufferless commits (bounded
        // by `cap_surface_damage`) so damage posted before a subsequent
        // same-size `NewBuffer` commit is included in its incremental row-bands.
        None => {
            if connected && state.frame_backpressure.is_deferred(surface_id) {
                if !surface_attributes.damage.is_empty() {
                    if let Some(BufferAssignment::New(metadata)) = tracked.state.buffer {
                        let rects = compositor_utils::damage_to_buffer_rects(
                            &surface_attributes.damage,
                            tracked.state.buffer_scale,
                            tracked
                                .state
                                .buffer_transform
                                .unwrap_or(Transform::Normal)
                                .into(),
                            tracked.state.viewport_state.as_ref(),
                            (metadata.width, metadata.height),
                        );
                        tracked.pending_damage.merge(rects);
                    } else {
                        tracked.pending_damage.mark_full();
                    }
                    surface_attributes.damage.clear();
                }
                tracked.has_committed = true;
                tracked.dirty = true;
                state.hold_frame_callbacks(surface, surface_id, frame_callbacks, frame_ready_at);
                return Ok(false);
            }
            state.schedule_or_fire_frame_callbacks(
                surface.clone(),
                surface_id,
                frame_callbacks,
                frame_ready_at,
            );
        },
    }

    let unmapped_xdg_surface = matches!(buffer_to_send, Some(BufferAssignment::Removed))
        && matches!(
            tracked.state.role,
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
    surface_state_to_send.buffer_delta = tracked.pending_buffer_delta.take();
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
                    if tracked.state.buffer.is_some()
                        && !self.frame_backpressure.is_deferred(surface_id)
                    {
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
    use std::time::Duration;

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
        frame_callbacks_done: Vec<u32>,
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

    impl Dispatch<wl_callback::WlCallback, u32> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &wl_callback::WlCallback,
            event: wl_callback::Event,
            tag: &u32,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if matches!(event, wl_callback::Event::Done { .. }) {
                state.frame_callbacks_done.push(*tag);
            }
        }
    }

    struct WaylandTestHarness {
        display: Display<WprsServerState>,
        event_loop: EventLoop<'static, WprsServerState>,
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
                event_loop,
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
                self.state.handle_event(conn_id, RecvType::Disconnected);
            }
            let (conn_id, rx) = self.state.serializer.connect_test_client();
            self.conn_id = Some(conn_id);
            self.rx = Some(rx);
            self.state.handle_event(conn_id, RecvType::Connected);
        }

        fn disconnect_wprsc(&mut self) {
            if let Some(conn_id) = self.conn_id.take() {
                self.state.serializer.disconnect_test_client(conn_id);
                self.state.handle_event(conn_id, RecvType::Disconnected);
            }
            self.rx = None;
        }

        fn ack_frame(&mut self, seq: u64) {
            let conn_id = self.conn_id.expect("wprsc must be connected");
            self.state
                .handle_event(conn_id, RecvType::Object(Event::FrameAck { seq }));
        }

        fn pump_timers(&mut self, duration: Duration) {
            let deadline = Instant::now() + duration;
            while Instant::now() < deadline {
                let remaining = deadline.saturating_duration_since(Instant::now());
                self.event_loop
                    .dispatch(
                        Some(remaining.min(Duration::from_millis(5))),
                        &mut self.state,
                    )
                    .unwrap();
                self.display.flush_clients().unwrap();
            }
            self.roundtrip();
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

        fn drain_messages(&self) -> Vec<SendType<Request>> {
            let Some(rx) = &self.rx else {
                return Vec::new();
            };
            let mut out = Vec::new();
            while let Ok(msg) = rx.try_recv() {
                out.push(msg);
            }
            out
        }

        fn drain_requests(&self) -> Vec<Request> {
            self.drain_messages()
                .into_iter()
                .filter_map(|msg| match msg {
                    SendType::Object(req) => Some(req),
                    SendType::RawBuffer(_) => None,
                })
                .collect()
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
            buffer_delta: None,
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

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_subsurface_parent_z_order_and_nested_subsurfaces_wire() {
        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let subcompositor = h.client_data.subcompositor.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();

        let p = compositor.create_surface(&h.qh, ());
        let p_xdg = wm_base.get_xdg_surface(&p, &h.qh, ());
        let _p_toplevel = p_xdg.get_toplevel(&h.qh, ());
        let c1 = compositor.create_surface(&h.qh, ());
        let c2 = compositor.create_surface(&h.qh, ());
        let c3 = compositor.create_surface(&h.qh, ());
        let gc = compositor.create_surface(&h.qh, ());

        let sub_c1 = subcompositor.get_subsurface(&c1, &p, &h.qh, ());
        sub_c1.set_position(10, 20);
        let sub_c2 = subcompositor.get_subsurface(&c2, &p, &h.qh, ());
        sub_c2.set_position(30, 40);
        let sub_c3 = subcompositor.get_subsurface(&c3, &p, &h.qh, ());
        sub_c3.set_position(50, 60);
        // Nested grandchild under `c1` must NOT appear in `p`'s direct z-order.
        let sub_gc = subcompositor.get_subsurface(&gc, &c1, &h.qh, ());
        sub_gc.set_position(7, 8);

        gc.commit();
        c1.commit();
        c2.commit();
        c3.commit();
        p.commit();
        h.roundtrip();

        let p_id = h.server_surface_id(&p);
        let c1_id = h.server_surface_id(&c1);
        let c2_id = h.server_surface_id(&c2);
        let c3_id = h.server_surface_id(&c3);
        let gc_id = h.server_surface_id(&gc);
        let sp = |id, x, y| SubsurfacePosition {
            id,
            position: (x, y).into(),
        };

        let last_z_order = |reqs: &[Request], target_sid: WlSurfaceId| {
            reqs.iter()
                .rev()
                .find_map(|req| match req {
                    Request::Surface(SurfaceRequest {
                        surface,
                        payload: SurfaceRequestPayload::Commit(state),
                        ..
                    }) if *surface == target_sid => Some(state.z_ordered_children.clone()),
                    _ => None,
                })
                .expect("expected Commit for surface")
        };

        // 1. Default stacking order: new subsurfaces stack above parent bottom-to-top.
        let reqs = h.drain_requests();
        assert_eq!(
            last_z_order(&reqs, p_id),
            vec![
                sp(p_id, 0, 0),
                sp(c1_id, 10, 20),
                sp(c2_id, 30, 40),
                sp(c3_id, 50, 60),
            ]
        );
        assert_eq!(
            last_z_order(&reqs, c1_id),
            vec![sp(c1_id, 0, 0), sp(gc_id, 7, 8)]
        );

        // 2. Place `c1` below `p`, `c2` below `c1`, and `gc` below `c1`.
        sub_c1.place_below(&p);
        sub_c2.place_below(&c1);
        sub_gc.place_below(&c1);
        c1.commit();
        p.commit();
        h.roundtrip();

        let reqs = h.drain_requests();
        assert_eq!(
            last_z_order(&reqs, p_id),
            vec![
                sp(c2_id, 30, 40),
                sp(c1_id, 10, 20),
                sp(p_id, 0, 0),
                sp(c3_id, 50, 60),
            ]
        );
        assert_eq!(
            last_z_order(&reqs, c1_id),
            vec![sp(gc_id, 7, 8), sp(c1_id, 0, 0)]
        );

        // 3. Reorder siblings across parent: `c3` below `p`, `c1` above `p` ->
        //    `[c2, c3, p, c1]`.
        sub_c3.place_below(&p);
        sub_c1.place_above(&p);
        p.commit();
        h.roundtrip();

        let reqs = h.drain_requests();
        let expected_p_order = vec![
            sp(c2_id, 30, 40),
            sp(c3_id, 50, 60),
            sp(p_id, 0, 0),
            sp(c1_id, 10, 20),
        ];
        assert_eq!(last_z_order(&reqs, p_id), expected_p_order);

        // 4. Reconnect `sync_surface_tree` preserves exact stacking order for
        //    both parent and nested child without mixing grandchildren.
        h.connect_wprsc();
        let sync_reqs = h.drain_requests();
        assert_eq!(last_z_order(&sync_reqs, p_id), expected_p_order);
        assert_eq!(
            last_z_order(&sync_reqs, c1_id),
            vec![sp(gc_id, 7, 8), sp(c1_id, 0, 0)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_toplevel_show_window_menu_forwards_mapped_serial_and_skips_unmapped_wire() {
        use crate::serialization::wayland::PointerEvent;
        use crate::serialization::wayland::PointerEventKind;

        let mut h = WaylandTestHarness::new(true);
        let conn_id = h.conn_id.unwrap();
        let compositor = h.client_data.compositor.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();
        let seat = h.client_data.seat.clone().unwrap();

        let surface = compositor.create_surface(&h.qh, ());
        let xdg_surf = wm_base.get_xdg_surface(&surface, &h.qh, ());
        let toplevel = xdg_surf.get_toplevel(&h.qh, ());
        h.roundtrip();
        let sid = h.server_surface_id(&surface);

        // Deliver a pointer button press with host serial 777 so `serial_map`
        // maps the client's server serial back to 777.
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::PointerFrame(vec![
                PointerEvent {
                    surface_id: sid,
                    position: (120.0, 45.0).into(),
                    kind: PointerEventKind::Enter { serial: 776 },
                },
                PointerEvent {
                    surface_id: sid,
                    position: (120.0, 45.0).into(),
                    kind: PointerEventKind::Press {
                        button: 0x111,
                        serial: 777,
                    },
                },
            ])),
        );
        h.roundtrip();
        let server_button_serial = h
            .client_data
            .last_button_serial
            .expect("client must receive wl_pointer.button serial");

        // 1. Pre-commit `show_window_menu` is suppressed by `committed_surfaces`.
        toplevel.show_window_menu(&seat, server_button_serial, 120, 45);
        h.roundtrip();
        assert!(h.drain_requests().is_empty());

        // 2. Post-commit `show_window_menu` forwards mapped serial 777 + coords.
        surface.commit();
        h.roundtrip();
        h.drain_requests();

        toplevel.show_window_menu(&seat, server_button_serial, 120, 45);
        h.roundtrip();
        let reqs = h.drain_requests();
        assert!(
            reqs.iter().any(|r| matches!(
                r,
                Request::Toplevel(ToplevelRequest {
                    surface,
                    payload: ToplevelRequestPayload::ShowWindowMenu {
                        serial: 777,
                        location,
                    },
                    ..
                }) if *surface == sid && *location == (120, 45).into()
            )),
            "expected ShowWindowMenu with serial 777 and location (120, 45), got {reqs:?}"
        );

        // 3. Unmapped serial is ignored.
        toplevel.show_window_menu(&seat, 888_888, 10, 20);
        h.roundtrip();
        assert!(h.drain_requests().is_empty());

        // 4. Stale serial from prior session is ignored after reconnect.
        h.connect_wprsc();
        h.drain_requests();
        toplevel.show_window_menu(&seat, server_button_serial, 30, 40);
        h.roundtrip();
        assert!(h.drain_requests().is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_buffer_delta_forwarded_once_and_not_replayed_wire() {
        use std::io::Seek;
        use std::io::Write;

        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();

        let surface = compositor.create_surface(&h.qh, ());
        let shm_path = std::env::temp_dir().join(format!(
            "wprs_delta_shm_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut shm_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&shm_path)
            .unwrap();
        shm_file.write_all(&[0x11; 16]).unwrap();
        _ = std::fs::remove_file(&shm_path);
        let pool = shm.create_pool(shm_file.as_fd(), 16, &h.qh, ());
        let buf = pool.create_buffer(0, 2, 2, 8, wl_shm::Format::Argb8888, &h.qh, ());

        // 1. Commit with non-zero `wl_surface.offset(-12, 7)` forwards `buffer_delta`.
        surface.attach(Some(&buf), 0, 0);
        surface.offset(-12, 7);
        surface.commit();
        h.roundtrip();
        let sid = h.server_surface_id(&surface);
        let reqs = h.drain_requests();
        let first_commit = reqs
            .iter()
            .find_map(|r| match r {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                }) if *surface == sid => Some(st),
                _ => None,
            })
            .expect("first commit must be forwarded");
        assert_eq!(first_commit.buffer_delta, Some((-12, 7).into()));

        // 2. Next commit with updated pixels and NO offset must NOT replay `(-12, 7)`.
        shm_file.rewind().unwrap();
        shm_file.write_all(&[0x22; 16]).unwrap();
        surface.attach(Some(&buf), 0, 0);
        surface.commit();
        h.roundtrip();
        let reqs = h.drain_requests();
        let second_commit = reqs
            .iter()
            .find_map(|r| match r {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                }) if *surface == sid => Some(st),
                _ => None,
            })
            .expect("second commit with new pixels must be forwarded");
        assert_eq!(second_commit.buffer_delta, None);

        // 3. Unchanged-pixel commit with a non-zero offset is NOT deduplicated away.
        surface.attach(Some(&buf), 0, 0);
        surface.offset(4, -3);
        surface.commit();
        h.roundtrip();
        let reqs = h.drain_requests();
        let third_commit = reqs
            .iter()
            .find_map(|r| match r {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                }) if *surface == sid => Some(st),
                _ => None,
            })
            .expect("offset-only commit must not be deduplicated");
        assert_eq!(third_commit.buffer, None);
        assert_eq!(third_commit.buffer_delta, Some((4, -3).into()));

        // 4. Subsequent zero-offset / delta-less bufferless commit is deduplicated.
        surface.offset(0, 0);
        surface.commit();
        h.roundtrip();
        assert!(h.drain_requests().is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_fullscreen_output_forwarding_and_configure_state_wire() {
        use crate::serialization::wayland::Mode as SerMode;
        use crate::serialization::wayland::OutputEvent;
        use crate::serialization::wayland::OutputInfo;
        use crate::serialization::wayland::Subpixel as SerSubpixel;
        use crate::serialization::wayland::Transform as SerTransform;

        let mut h = WaylandTestHarness::new(true);
        let conn_id = h.conn_id.unwrap();

        let make_output = |id: u32, name: &str, x: i32| OutputInfo {
            id,
            model: format!("Model-{id}"),
            make: "Vendor".to_string(),
            location: (x, 0).into(),
            physical_size: (600, 340).into(),
            subpixel: SerSubpixel::HorizontalRgb,
            transform: SerTransform::Normal,
            scale_factor: 1,
            mode: SerMode {
                dimensions: (1920, 1080).into(),
                refresh_rate: 60000,
                current: true,
                preferred: true,
            },
            logical_position: Some((x, 0).into()),
            logical_size: Some((1920, 1080).into()),
            name: Some(name.to_string()),
            description: None,
        };

        for output in [make_output(10, "DP-1", 0), make_output(20, "DP-2", 1920)] {
            h.state.handle_event(
                conn_id,
                RecvType::Object(Event::Output(OutputEvent::New(output))),
            );
        }
        h.roundtrip();
        h.roundtrip();

        let find_client_output = |h: &WaylandTestHarness, suffix: &str| {
            let global_name = *h
                .client_data
                .output_snapshots
                .iter()
                .find_map(|(name, snap)| {
                    snap.wl_name
                        .as_deref()
                        .is_some_and(|n| n.ends_with(suffix))
                        .then_some(name)
                })
                .expect("client must bind output global");
            h.client_data.wl_outputs[&global_name].clone()
        };
        let out_10 = find_client_output(&h, "_DP-1");
        let out_20 = find_client_output(&h, "_DP-2");

        let compositor = h.client_data.compositor.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();
        let surface = compositor.create_surface(&h.qh, ());
        let xdg_surf = wm_base.get_xdg_surface(&surface, &h.qh, ());
        let toplevel = xdg_surf.get_toplevel(&h.qh, ());

        // 1. Pre-commit `set_fullscreen(Some(out_20))` records `FullscreenState::Set(Some(20))`.
        toplevel.set_fullscreen(Some(&out_20));
        surface.commit();
        h.roundtrip();
        let sid = h.server_surface_id(&surface);
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == sid
                && matches!(
                    &st.role,
                    Some(Role::XdgToplevel(tl)) if tl.fullscreen == Some(FullscreenState::Set(Some(20)))
                )
        )));

        // 2. Post-commit `set_fullscreen(Some(out_10))` sends `SetFullscreen(Some(10))`.
        toplevel.set_fullscreen(Some(&out_10));
        h.roundtrip();
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Toplevel(ToplevelRequest {
                surface,
                payload: ToplevelRequestPayload::SetFullscreen(Some(10)),
                ..
            }) if *surface == sid
        )));

        // 3. Host fullscreen configure populates `state.fullscreen_output` on Smithay's
        //    pending toplevel state, and non-fullscreen configure clears it.
        let mut fs_configure =
            smithay_client_toolkit::shell::xdg::window::WindowConfigure::default();
        fs_configure.state = smithay_client_toolkit::reexports::csd_frame::WindowState::FULLSCREEN
            | smithay_client_toolkit::reexports::csd_frame::WindowState::ACTIVATED;
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Toplevel(ToplevelEvent::Configure(
                ToplevelConfigure::from_smithay(&sid, fs_configure),
            ))),
        );
        let server_tl = h
            .state
            .xdg_shell_state
            .toplevel_surfaces()
            .iter()
            .find(|tl| WlSurfaceId::new(tl.wl_surface()) == sid)
            .unwrap()
            .clone();
        let configured_fs_out = compositor::with_states(server_tl.wl_surface(), |data| {
            data.data_map
                .get::<XdgToplevelSurfaceData>()
                .unwrap()
                .lock()
                .unwrap()
                .current_server_state()
                .fullscreen_output
                .clone()
        });
        assert_eq!(
            configured_fs_out
                .as_ref()
                .and_then(|o| h.state.output_id_for_wl_output(o)),
            Some(10)
        );

        let mut normal_configure =
            smithay_client_toolkit::shell::xdg::window::WindowConfigure::default();
        normal_configure.state =
            smithay_client_toolkit::reexports::csd_frame::WindowState::ACTIVATED;
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Toplevel(ToplevelEvent::Configure(
                ToplevelConfigure::from_smithay(&sid, normal_configure),
            ))),
        );
        let cleared_fs_out = compositor::with_states(server_tl.wl_surface(), |data| {
            data.data_map
                .get::<XdgToplevelSurfaceData>()
                .unwrap()
                .lock()
                .unwrap()
                .current_server_state()
                .fullscreen_output
                .clone()
        });
        assert!(cleared_fs_out.is_none());

        // 4. Reconnect clears stale `output_id` in `FullscreenState::Set` so the
        //    new session does not bind to an unrelated output ID.
        h.connect_wprsc();
        let sync_reqs = h.drain_requests();
        assert!(sync_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == sid
                && matches!(
                    &st.role,
                    Some(Role::XdgToplevel(tl)) if tl.fullscreen == Some(FullscreenState::Set(None))
                )
        )));

        // 5. `set_fullscreen(None)` and `unset_fullscreen()` forward cleanly.
        toplevel.set_fullscreen(None);
        toplevel.unset_fullscreen();
        h.roundtrip();
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Toplevel(ToplevelRequest {
                surface,
                payload: ToplevelRequestPayload::SetFullscreen(None),
                ..
            }) if *surface == sid
        )));
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Toplevel(ToplevelRequest {
                surface,
                payload: ToplevelRequestPayload::UnsetFullscreen,
                ..
            }) if *surface == sid
        )));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_frame_backpressure_coalescing_fair_fifo_pixels_and_reconnect_wire() {
        use std::io::Seek;
        use std::io::Write;
        use std::num::NonZeroUsize;

        use crate::filtering::ValidatedTileStream;
        use crate::serialization::framing::CompressedTileStream;
        use crate::sharding_compression::ShardingDecompressor;

        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let subcompositor = h.client_data.subcompositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();

        let width = 16usize;
        let height = 16usize;
        let stride = width * 4;
        let byte_len = height * stride;

        let make_shm_surface = |tag: &str, h: &mut WaylandTestHarness| {
            let path = std::env::temp_dir().join(format!(
                "wprs_bp_test_{tag}_{}_{}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            let pixels = vec![0x10u8; byte_len];
            file.write_all(&pixels).unwrap();
            _ = std::fs::remove_file(&path);
            let pool = shm.create_pool(file.as_fd(), byte_len as i32, &h.qh, ());
            let buf = pool.create_buffer(
                0,
                width as i32,
                height as i32,
                stride as i32,
                wl_shm::Format::Argb8888,
                &h.qh,
                (),
            );
            let surf = compositor.create_surface(&h.qh, ());
            (surf, pool, buf, file, pixels)
        };

        let paint_rect =
            |pixels: &mut [u8], x: usize, y: usize, w: usize, hh: usize, px: [u8; 4]| {
                for row in y..(y + hh) {
                    for col in x..(x + w) {
                        let off = row * stride + col * 4;
                        pixels[off..off + 4].copy_from_slice(&px);
                    }
                }
            };

        let mut decompressor = ShardingDecompressor::new(NonZeroUsize::new(2).unwrap()).unwrap();
        let mut tile_cache = filtering::ClientTileCache::new();
        let mut s1_canvas = vec![0u8; byte_len];
        let mut s2_canvas = vec![0u8; byte_len];
        let decode_into = |streaming: crate::serialization::framing::StreamingCompressedTiles,
                           canvas: &mut [u8],
                           decompressor: &mut ShardingDecompressor,
                           tile_cache: &mut filtering::ClientTileCache|
         -> u64 {
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
            let seq = stream.header.frame_seq;
            let validated = ValidatedTileStream::new(stream).unwrap();
            filtering::decompress_and_unfilter_streaming(
                validated,
                decompressor,
                canvas,
                Some(tile_cache),
            )
            .unwrap();
            reader.join().unwrap();
            seq
        };

        let (s1, _p1, b1, mut f1, mut px1) = make_shm_surface("s1", &mut h);
        let (s2, _p2, b2, mut f2, mut px2) = make_shm_surface("s2", &mut h);

        // 1. First `MAX_IN_FLIGHT_FRAMES` commits consume all credits immediately.
        let mut in_flight_seqs = Vec::new();
        for step in 0..constants::MAX_IN_FLIGHT_FRAMES {
            paint_rect(
                &mut px1,
                0,
                0,
                width,
                height,
                [0x20 + step as u8, 0x30, 0x40, 0xFF],
            );
            f1.rewind().unwrap();
            f1.write_all(&px1).unwrap();
            s1.attach(Some(&b1), 0, 0);
            s1.commit();
            h.roundtrip();
        }
        let sid1 = h.server_surface_id(&s1);
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let seq = decode_into(rb, &mut s1_canvas, &mut decompressor, &mut tile_cache);
                in_flight_seqs.push(seq);
            }
        }
        assert_eq!(in_flight_seqs.len(), constants::MAX_IN_FLIGHT_FRAMES);
        assert_eq!(
            h.state.frame_backpressure.in_flight_count(),
            constants::MAX_IN_FLIGHT_FRAMES
        );
        assert_eq!(h.state.frame_backpressure.deferred_len(), 0);
        assert_eq!(s1_canvas, px1);

        // 2. Rapidly commit 12 more frames on S1 (with disjoint partial damage rects
        //    and accumulated buffer_delta) interleaved with commits on S2 without
        //    any FrameAck -> zero extra RawBuffers sent, both coalesced in FIFO order!
        for i in 0..6u8 {
            paint_rect(&mut px1, 0, 0, 8, 8, [0xA0 + i, 0x11, 0x22, 0xFF]);
            f1.rewind().unwrap();
            f1.write_all(&px1).unwrap();
            s1.attach(Some(&b1), 0, 0);
            s1.damage_buffer(0, 0, 8, 8);
            if i == 0 {
                s1.offset(3, -1);
            }
            s1.commit();

            paint_rect(&mut px2, 0, 0, width, height, [0x50 + i, 0x60, 0x70, 0xFF]);
            f2.rewind().unwrap();
            f2.write_all(&px2).unwrap();
            s2.attach(Some(&b2), 0, 0);
            s2.commit();

            paint_rect(&mut px1, 8, 8, 8, 8, [0xB0 + i, 0x33, 0x44, 0xFF]);
            f1.rewind().unwrap();
            f1.write_all(&px1).unwrap();
            s1.attach(Some(&b1), 0, 0);
            s1.damage_buffer(8, 8, 8, 8);
            if i == 5 {
                s1.offset(-1, 5);
            }
            s1.commit();
            h.roundtrip();
        }
        let sid2 = h.server_surface_id(&s2);
        assert!(
            h.drain_messages().is_empty(),
            "no frames or commits should be emitted while credits are exhausted"
        );
        assert_eq!(
            h.state.frame_backpressure.in_flight_count(),
            constants::MAX_IN_FLIGHT_FRAMES
        );
        assert_eq!(h.state.frame_backpressure.deferred_len(), 2);

        // 3. Stale/unknown FrameAck is ignored and does not flush anything.
        h.ack_frame(99_999);
        assert!(h.drain_messages().is_empty());
        assert_eq!(
            h.state.frame_backpressure.in_flight_count(),
            constants::MAX_IN_FLIGHT_FRAMES
        );

        // 4. Ack 1st credit -> S1 flushes FIRST (FIFO) with exact coalesced pixels
        //    from BOTH partial damage rects and summed `buffer_delta == (2, 4)`.
        h.ack_frame(in_flight_seqs.remove(0));
        let msgs = h.drain_messages();
        let mut s1_flushed_seq = None;
        let mut s1_flushed_delta = None;
        for msg in msgs {
            match msg {
                SendType::RawBuffer(rb) => {
                    assert_eq!(rb.header.surface, sid1);
                    s1_flushed_seq = Some(decode_into(
                        rb,
                        &mut s1_canvas,
                        &mut decompressor,
                        &mut tile_cache,
                    ));
                },
                SendType::Object(Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                })) => {
                    assert_eq!(surface, sid1);
                    s1_flushed_delta = st.buffer_delta;
                },
                other => panic!("unexpected message on S1 flush: {other:?}"),
            }
        }
        in_flight_seqs.push(s1_flushed_seq.expect("S1 must flush RawBuffer"));
        assert_eq!(s1_flushed_delta, Some((2, 4).into()));
        assert_eq!(
            s1_canvas, px1,
            "coalesced S1 pixels must match final frame bit-for-bit"
        );
        assert_eq!(h.state.frame_backpressure.deferred_len(), 1);

        // 5. Ack 2nd credit -> S2 flushes NEXT (fair FIFO!) with exact final pixels.
        h.ack_frame(in_flight_seqs.remove(0));
        let msgs = h.drain_messages();
        let mut s2_flushed_seq = None;
        for msg in msgs {
            if let SendType::RawBuffer(rb) = msg {
                assert_eq!(rb.header.surface, sid2);
                s2_flushed_seq = Some(decode_into(
                    rb,
                    &mut s2_canvas,
                    &mut decompressor,
                    &mut tile_cache,
                ));
            }
        }
        in_flight_seqs.push(s2_flushed_seq.expect("S2 must flush RawBuffer"));
        assert_eq!(
            s2_canvas, px2,
            "coalesced S2 pixels must match final frame bit-for-bit"
        );
        assert_eq!(h.state.frame_backpressure.deferred_len(), 0);

        // 6. Unmap (`attach(None)`) on a deferred surface removes it from deferred
        //    queue and immediately forwards `BufferAssignment::Removed`.
        paint_rect(&mut px1, 0, 0, width, height, [0xCC, 0xCC, 0xCC, 0xFF]);
        f1.rewind().unwrap();
        f1.write_all(&px1).unwrap();
        s1.attach(Some(&b1), 0, 0);
        s1.commit();
        h.roundtrip();
        assert!(h.state.frame_backpressure.is_deferred(sid1));
        assert!(h.drain_messages().is_empty());

        s1.attach(None, 0, 0);
        s1.commit();
        h.roundtrip();
        assert!(!h.state.frame_backpressure.is_deferred(sid1));
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == sid1 && matches!(st.buffer, Some(BufferAssignment::Removed))
        )));

        // 7. Destroying a deferred surface removes it cleanly from deferred queue.
        paint_rect(&mut px2, 0, 0, width, height, [0xDD, 0xDD, 0xDD, 0xFF]);
        f2.rewind().unwrap();
        f2.write_all(&px2).unwrap();
        s2.attach(Some(&b2), 0, 0);
        s2.commit();
        h.roundtrip();
        assert!(h.state.frame_backpressure.is_deferred(sid2));
        s2.destroy();
        h.roundtrip();
        assert!(!h.state.frame_backpressure.is_deferred(sid2));
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Destroyed,
                ..
            }) if *surface == sid2
        )));

        // 8. Deferred sync subsurface: parent commits while sync child is deferred,
        //    then acking a credit flushes child AND emits a bufferless parent Commit
        //    so `wprsc` applies the sync child.
        let parent = compositor.create_surface(&h.qh, ());
        let (child, _pc, bc, mut fc, mut pxc) = make_shm_surface("sync_child", &mut h);
        let _sub = subcompositor.get_subsurface(&child, &parent, &h.qh, ());
        paint_rect(&mut pxc, 0, 0, width, height, [0x77, 0x88, 0x99, 0xFF]);
        fc.rewind().unwrap();
        fc.write_all(&pxc).unwrap();
        child.attach(Some(&bc), 0, 0);
        child.commit();
        parent.commit();
        h.roundtrip();
        let pid = h.server_surface_id(&parent);
        let cid = h.server_surface_id(&child);
        assert!(h.state.frame_backpressure.is_deferred(cid));
        assert!(h.state.deferred_sync_ancestors.contains(&pid));
        h.drain_messages();

        h.ack_frame(in_flight_seqs.remove(0));
        let msgs = h.drain_messages();
        let mut saw_child_buf = false;
        let mut saw_child_commit = false;
        let mut saw_parent_flush_commit = false;
        for msg in msgs {
            match msg {
                SendType::RawBuffer(rb) if rb.header.surface == cid => {
                    saw_child_buf = true;
                    in_flight_seqs.push(rb.header.frame_seq);
                },
                SendType::Object(Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(_),
                    ..
                })) if surface == cid => {
                    saw_child_commit = true;
                },
                SendType::Object(Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                })) if surface == pid && st.buffer.is_none() => {
                    saw_parent_flush_commit = true;
                },
                _ => {},
            }
        }
        assert!(saw_child_buf && saw_child_commit && saw_parent_flush_commit);
        assert!(!h.state.deferred_sync_ancestors.contains(&pid));

        // 9. Reconnect resets credits and caps initial `sync_surface_tree` across
        //    `MAX_IN_FLIGHT_FRAMES + 2` buffered toplevels.
        h.disconnect_wprsc();
        let mut toplevels = Vec::new();
        for idx in 0..(constants::MAX_IN_FLIGHT_FRAMES + 2) {
            let (surf, pool, buf, mut file, mut px) =
                make_shm_surface(&format!("tl_{idx}"), &mut h);
            let xdg_surf = wm_base.get_xdg_surface(&surf, &h.qh, ());
            let tl = xdg_surf.get_toplevel(&h.qh, ());
            surf.commit();
            h.roundtrip();
            paint_rect(
                &mut px,
                0,
                0,
                width,
                height,
                [0x10 + idx as u8, 0x20, 0x30, 0xFF],
            );
            file.rewind().unwrap();
            file.write_all(&px).unwrap();
            surf.attach(Some(&buf), 0, 0);
            surf.commit();
            toplevels.push((surf, xdg_surf, tl, pool, buf, file));
        }
        h.roundtrip();

        h.connect_wprsc();
        let sync_msgs = h.drain_messages();
        let mut sync_seqs: Vec<u64> = sync_msgs
            .iter()
            .filter_map(|m| match m {
                SendType::RawBuffer(rb) => Some(rb.header.frame_seq),
                SendType::Object(_) => None,
            })
            .collect();
        assert_eq!(sync_seqs.len(), constants::MAX_IN_FLIGHT_FRAMES);
        assert_eq!(
            h.state.frame_backpressure.in_flight_count(),
            constants::MAX_IN_FLIGHT_FRAMES
        );
        assert_eq!(h.state.frame_backpressure.deferred_len(), 2);

        // Acking two synced frames flushes the remaining 2 deferred toplevels.
        for _ in 0..2 {
            h.ack_frame(sync_seqs.remove(0));
            let flushed_rbs: Vec<u64> = h
                .drain_messages()
                .into_iter()
                .filter_map(|m| match m {
                    SendType::RawBuffer(rb) => Some(rb.header.frame_seq),
                    SendType::Object(_) => None,
                })
                .collect();
            assert_eq!(flushed_rbs.len(), 1);
            sync_seqs.extend(flushed_rbs);
        }
        assert_eq!(h.state.frame_backpressure.deferred_len(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wl_surface_frame_pacing_deferred_detached_and_disconnect_wire() {
        use std::io::Seek;
        use std::io::Write;

        let mut h = WaylandTestHarness::new(true);
        // Use a short frame interval so timer tests finish in milliseconds.
        h.state.frame_interval = Duration::from_millis(5);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();

        let path = std::env::temp_dir().join(format!(
            "wprs_frame_pace_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&[0x11; 16]).unwrap();
        _ = std::fs::remove_file(&path);
        let pool = shm.create_pool(file.as_fd(), 16, &h.qh, ());
        let buf = pool.create_buffer(0, 2, 2, 8, wl_shm::Format::Argb8888, &h.qh, ());
        let surface = compositor.create_surface(&h.qh, ());

        let mut in_flight_seqs = Vec::new();
        let commit_frame =
            |h: &mut WaylandTestHarness, file: &mut std::fs::File, fill: u8, cb_tag: u32| {
                file.rewind().unwrap();
                file.write_all(&[fill; 16]).unwrap();
                surface.attach(Some(&buf), 0, 0);
                let _cb = surface.frame(&h.qh, cb_tag);
                surface.commit();
                h.roundtrip();
            };

        // 1. Unthrottled commits consume credits and fire `wl_surface.frame` callbacks.
        for i in 0..constants::MAX_IN_FLIGHT_FRAMES {
            commit_frame(&mut h, &mut file, 0x20 + i as u8, 100 + i as u32);
        }
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                in_flight_seqs.push(rb.header.frame_seq);
            }
        }
        assert_eq!(in_flight_seqs.len(), constants::MAX_IN_FLIGHT_FRAMES);
        h.pump_timers(Duration::from_millis(20));
        assert_eq!(h.client_data.frame_callbacks_done, vec![100, 101, 102]);
        h.client_data.frame_callbacks_done.clear();

        // 2. Backpressured commit is deferred and HOLDS its `wl_surface.frame` callback.
        commit_frame(&mut h, &mut file, 0x40, 200);
        assert!(h.drain_messages().is_empty());
        h.pump_timers(Duration::from_millis(25));
        assert!(
            h.client_data.frame_callbacks_done.is_empty(),
            "frame callback must remain held while surface is deferred"
        );

        // 3. Acking a credit flushes the surface and releases its held frame callback.
        h.ack_frame(in_flight_seqs.remove(0));
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                in_flight_seqs.push(rb.header.frame_seq);
            }
        }
        h.pump_timers(Duration::from_millis(20));
        assert_eq!(h.client_data.frame_callbacks_done, vec![200]);
        h.client_data.frame_callbacks_done.clear();

        // 4. Disconnecting while a surface is deferred releases its held callback
        //    so apps never freeze on detach.
        commit_frame(&mut h, &mut file, 0x50, 300);
        h.pump_timers(Duration::from_millis(20));
        assert!(h.client_data.frame_callbacks_done.is_empty());

        h.disconnect_wprsc();
        h.pump_timers(Duration::from_millis(20));
        assert_eq!(h.client_data.frame_callbacks_done, vec![300]);
        h.client_data.frame_callbacks_done.clear();

        // 5. Detached commits fire `wl_surface.frame` normally.
        commit_frame(&mut h, &mut file, 0x60, 400);
        h.pump_timers(Duration::from_millis(20));
        assert_eq!(h.client_data.frame_callbacks_done, vec![400]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsd_scroll_blit_commit_damage_and_bit_exact_decode_wire() {
        use std::io::Seek;
        use std::io::Write;
        use std::num::NonZeroUsize;

        use crate::filtering::ValidatedTileStream;
        use crate::serialization::framing::CompressedTileStream;
        use crate::sharding_compression::ShardingDecompressor;

        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();

        let width = 640usize;
        let height = 512usize;
        let stride = width * 4;
        let byte_len = height * stride;

        let mut doc = vec![0u8; (height + 200) * stride];
        for y in 0..(height + 200) {
            let line = y / 16;
            for x in 0..width {
                let off = y * stride + x * 4;
                let glyph = (3..=12).contains(&(y % 16))
                    && x >= 16
                    && x < width - 16
                    && (((x * 29) ^ (y * 17) ^ (line * 83)) % 5) < 2;
                let px = if glyph {
                    [
                        ((line * 23 + x) & 0x7F) as u8,
                        ((line * 37 + y) & 0x7F) as u8,
                        ((line * 13 + x * 5) & 0x7F) as u8,
                        0xFF,
                    ]
                } else {
                    [246, 248, 250, 0xFF]
                };
                doc[off..off + 4].copy_from_slice(&px);
            }
        }

        let path = std::env::temp_dir().join(format!(
            "wprs_scroll_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&doc[100 * stride..100 * stride + byte_len])
            .unwrap();
        _ = std::fs::remove_file(&path);
        let pool = shm.create_pool(file.as_fd(), byte_len as i32, &h.qh, ());
        let buf = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Xrgb8888,
            &h.qh,
            (),
        );
        let surface = compositor.create_surface(&h.qh, ());
        let mut decompressor = ShardingDecompressor::new(NonZeroUsize::new(2).unwrap()).unwrap();
        let mut tile_cache = filtering::ClientTileCache::new();
        let mut canvas = vec![0u8; byte_len];

        let mut decode_rb = |rb: crate::serialization::framing::StreamingCompressedTiles,
                             canvas: &mut [u8]| {
            let mut wire = Vec::new();
            rb.framed_write(&mut wire).unwrap();
            let wire_len = wire.len();
            let (tx, rx) = crossbeam_channel::bounded(1);
            let reader = std::thread::spawn(move || {
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |stream| {
                    tx.send(stream).unwrap();
                    Ok(())
                })
                .unwrap();
            });
            let stream = rx.recv().unwrap();
            let seq = stream.header.frame_seq;
            let validated = ValidatedTileStream::new(stream).unwrap();
            let sr = validated.scroll_rect();
            let tiles = validated
                .incremental_tile_rects()
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            if let Some(sr) = sr {
                filtering::apply_scroll_rect_in_place(canvas, stride, sr);
            }
            filtering::decompress_and_unfilter_streaming(
                validated,
                &mut decompressor,
                canvas,
                Some(&mut tile_cache),
            )
            .unwrap();
            reader.join().unwrap();
            (seq, sr, tiles, wire_len)
        };

        // 1. Initial full commit.
        let mut scroll_y = 100usize;
        surface.attach(Some(&buf), 0, 0);
        surface.commit();
        h.roundtrip();
        let msgs = h.drain_messages();
        let mut full_wire_len = 0usize;
        for msg in msgs {
            if let SendType::RawBuffer(rb) = msg {
                let (seq, sr, _, wire_len) = decode_rb(rb, &mut canvas);
                assert_eq!(sr, None);
                full_wire_len = wire_len;
                h.ack_frame(seq);
            }
        }
        assert_eq!(canvas, doc[scroll_y * stride..scroll_y * stride + byte_len]);

        // 2. Scroll down (`dy = -48`) and scroll up (`dy = +32`) commits.
        for expected_dy in [-48i32, 32i32] {
            scroll_y = (scroll_y as isize - expected_dy as isize) as usize;
            let expected = &doc[scroll_y * stride..scroll_y * stride + byte_len];
            file.rewind().unwrap();
            file.write_all(expected).unwrap();
            surface.attach(Some(&buf), 0, 0);
            surface.damage_buffer(0, 0, width as i32, height as i32);
            surface.commit();
            h.roundtrip();

            let mut got_sr = None;
            let mut got_tiles = Vec::new();
            let mut got_damage = None;
            for msg in h.drain_messages() {
                match msg {
                    SendType::RawBuffer(rb) => {
                        let (seq, sr, tiles, wire_len) = decode_rb(rb, &mut canvas);
                        assert!(
                            wire_len * 3 < full_wire_len,
                            "expected >=3x wire reduction for dy={expected_dy}: {wire_len} vs {full_wire_len}"
                        );
                        got_sr = sr;
                        got_tiles = tiles;
                        h.ack_frame(seq);
                    },
                    SendType::Object(Request::Surface(SurfaceRequest {
                        payload: SurfaceRequestPayload::Commit(st),
                        ..
                    })) => {
                        got_damage = st.damage;
                    },
                    _ => {},
                }
            }
            let sr = got_sr.expect("wprsd must emit ScrollRect");
            assert_eq!(sr.dy, expected_dy);
            let dmg = got_damage.expect("wprsd must emit explicit damage list");
            assert!(dmg.contains(&sr.dst_damage_rect()));
            for t in &got_tiles {
                assert!(dmg.contains(&Rectangle::new(
                    t.x as i32, t.y as i32, t.w as i32, t.h as i32
                )));
            }
            assert_eq!(canvas, expected);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_wprsd_tile_cache_aba_resync_disconnect_and_reconnect_wire() {
        use std::io::Seek;
        use std::io::Write;
        use std::num::NonZeroUsize;

        use crate::filtering::ClientTileCache;
        use crate::filtering::ValidatedTileStream;
        use crate::serialization::framing::CompressedTile;
        use crate::serialization::framing::CompressedTileStream;
        use crate::serialization::framing::TilePayload;
        use crate::serialization::wayland::SurfaceEvent;
        use crate::serialization::wayland::SurfaceEventPayload;
        use crate::sharding_compression::ShardingDecompressor;

        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();

        let width = 512usize;
        let height = 256usize;
        let stride = width * 4;
        let byte_len = height * stride;

        let make_frame = |seed: u8| -> Vec<u8> {
            (0..byte_len)
                .map(|i| {
                    if i % 4 == 3 {
                        0xFF
                    } else {
                        seed.wrapping_mul(37).wrapping_add(((i * 13) % 251) as u8)
                    }
                })
                .collect()
        };
        let frame_a = make_frame(1);
        let frame_b = make_frame(2);

        let path = std::env::temp_dir().join(format!(
            "wprs_tile_cache_srv_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&frame_a).unwrap();
        _ = std::fs::remove_file(&path);
        let pool = shm.create_pool(file.as_fd(), byte_len as i32, &h.qh, ());
        let buf = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Xrgb8888,
            &h.qh,
            (),
        );
        let surface = compositor.create_surface(&h.qh, ());
        let mut decompressor = ShardingDecompressor::new(NonZeroUsize::new(2).unwrap()).unwrap();
        let mut client_cache = ClientTileCache::new();
        let mut canvas = vec![0u8; byte_len];

        let mut decode_rb = |rb: crate::serialization::framing::StreamingCompressedTiles,
                             canvas: &mut [u8],
                             cli: &mut ClientTileCache| {
            let mut wire = Vec::new();
            rb.framed_write(&mut wire).unwrap();
            let mut rx_opt = None;
            CompressedTileStream::read_from_stream(&mut wire.as_slice(), |stream| {
                rx_opt = Some(stream);
                Ok(())
            })
            .unwrap();
            let stream = rx_opt.unwrap();
            let header = stream.header.clone();
            let tiles: Vec<CompressedTile> = stream
                .tiles
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .unwrap();
            let n_hits = tiles
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                .count();
            let (tx, rx) = crossbeam_channel::unbounded();
            for t in tiles {
                tx.send(Ok(t)).unwrap();
            }
            drop(tx);
            let validated = ValidatedTileStream::new(CompressedTileStream {
                header: header.clone(),
                tiles: rx,
            })
            .unwrap();
            filtering::decompress_and_unfilter_streaming(
                validated,
                &mut decompressor,
                canvas,
                Some(cli),
            )
            .unwrap();
            (header, n_hits)
        };

        let mut commit_frame = |h: &mut WaylandTestHarness, pixels: &[u8]| {
            file.rewind().unwrap();
            file.write_all(pixels).unwrap();
            surface.attach(Some(&buf), 0, 0);
            surface.damage_buffer(0, 0, width as i32, height as i32);
            surface.commit();
            h.roundtrip();
        };

        // 1. Frame A -> first emitted header after connect sets `reset_tile_cache = true`.
        commit_frame(&mut h, &frame_a);
        let sid = h.server_surface_id(&surface);
        let mut got_a1 = None;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let (hdr, hits) = decode_rb(rb, &mut canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                got_a1 = Some((hdr, hits));
            }
        }
        let (hdr_a1, hits_a1) = got_a1.expect("expected RawBuffer for frame A1");
        assert!(hdr_a1.reset_tile_cache);
        assert_eq!(hits_a1, 0);
        assert_eq!(canvas, frame_a);

        // 2. Frame B -> Frame A: return-to-A hits 100% of cached tiles!
        commit_frame(&mut h, &frame_b);
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let (hdr, _) = decode_rb(rb, &mut canvas, &mut client_cache);
                assert!(!hdr.reset_tile_cache);
                h.ack_frame(hdr.frame_seq);
            }
        }
        assert_eq!(canvas, frame_b);

        commit_frame(&mut h, &frame_a);
        let mut got_a2 = None;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let (hdr, hits) = decode_rb(rb, &mut canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                got_a2 = Some((hdr, hits));
            }
        }
        let (hdr_a2, hits_a2) = got_a2.expect("expected RawBuffer for frame A2");
        assert!(!hdr_a2.reset_tile_cache);
        assert_eq!(hits_a2, hdr_a2.tile_rects.len());
        assert_eq!(canvas, frame_a);

        // 3. `BufferResyncRequested` resets server tile cache and sets `reset_tile_cache = true`
        //    on the next emitted full frame (self-healing, 0 CacheHits).
        let conn_id = h.state.active_conn_id.unwrap();
        h.state.handle_event(
            conn_id,
            crate::serialization::RecvType::Object(crate::serialization::Event::Surface(
                SurfaceEvent {
                    surface_id: sid,
                    payload: SurfaceEventPayload::BufferResyncRequested,
                },
            )),
        );
        assert_eq!(h.state.tile_cache.occupied_slots(), 0);
        commit_frame(&mut h, &frame_b);
        let mut got_resync = None;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let (hdr, hits) = decode_rb(rb, &mut canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                got_resync = Some((hdr, hits));
            }
        }
        let (hdr_resync, hits_resync) = got_resync.expect("expected RawBuffer after resync");
        assert!(hdr_resync.reset_tile_cache);
        assert!(!hdr_resync.incremental);
        assert_eq!(hits_resync, 0);
        assert_eq!(canvas, frame_b);

        // 4. Disconnect clears server tile cache; reconnect starts fresh with `reset_tile_cache = true`.
        assert!(h.state.tile_cache.occupied_slots() > 0);
        h.disconnect_wprsc();
        assert_eq!(h.state.tile_cache.occupied_slots(), 0);
        assert_eq!(h.state.tile_cache.used_bytes(), 0);

        h.connect_wprsc();
        client_cache.clear();
        let _ = h.drain_messages();
        commit_frame(&mut h, &frame_a);
        let mut got_reconn = None;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let (hdr, hits) = decode_rb(rb, &mut canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                got_reconn = Some((hdr, hits));
            }
        }
        let (hdr_reconn, hits_reconn) = got_reconn.expect("expected RawBuffer after reconnect");
        assert!(hdr_reconn.reset_tile_cache);
        assert_eq!(hits_reconn, 0);
        assert_eq!(canvas, frame_a);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dormant_shadow_compression_lifecycle_unchanged_wake_and_resize() {
        use std::io::Seek;
        use std::io::Write;
        use std::num::NonZeroUsize;

        use crate::filtering::ValidatedTileStream;
        use crate::serialization::framing::CompressedTileStream;
        use crate::sharding_compression::ShardingDecompressor;

        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();

        // 512x512 Xrgb8888 = 1,048,576 B == `MIN_DORMANT_COMPRESS_BYTES`.
        let width = 512usize;
        let height = 512usize;
        let stride = width * 4;
        let byte_len = height * stride;
        assert_eq!(byte_len, filtering::MIN_DORMANT_COMPRESS_BYTES);

        let path = std::env::temp_dir().join(format!(
            "wprs_dormant_shadow_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let mut frame = vec![0u8; byte_len];
        for y in 0..height {
            for x in 0..width {
                let off = y * stride + x * 4;
                frame[off] = ((x / 32) * 19 + (y / 32) * 7) as u8;
                frame[off + 1] = ((y / 16) * 11) as u8;
                frame[off + 2] = ((x / 64) * 23) as u8;
                frame[off + 3] = 0xFF;
            }
        }
        file.write_all(&frame).unwrap();
        _ = std::fs::remove_file(&path);
        let pool = shm.create_pool(file.as_fd(), byte_len as i32, &h.qh, ());
        let buf = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Xrgb8888,
            &h.qh,
            (),
        );
        let surf = compositor.create_surface(&h.qh, ());

        let mut decompressor = ShardingDecompressor::new(NonZeroUsize::new(2).unwrap()).unwrap();
        let mut client_cache = filtering::ClientTileCache::new();
        let mut canvas = vec![0u8; byte_len];
        let mut decode_rb = |rb: crate::serialization::framing::StreamingCompressedTiles,
                             dst: &mut [u8],
                             cache: &mut filtering::ClientTileCache|
         -> crate::serialization::framing::RawBufferHeader {
            let mut wire = Vec::new();
            rb.framed_write(&mut wire).unwrap();
            let mut cursor = wire.as_slice();
            let mut rx = None;
            CompressedTileStream::read_from_stream(&mut cursor, |s| {
                rx = Some(s);
                Ok(())
            })
            .unwrap();
            let stream = rx.unwrap();
            let hdr = stream.header.clone();
            let validated = ValidatedTileStream::new(stream).unwrap();
            filtering::decompress_and_unfilter_streaming(
                validated,
                &mut decompressor,
                dst,
                Some(cache),
            )
            .unwrap();
            hdr
        };

        // 1. Initial commit populates raw shadow + quadrant hashes.
        surf.attach(Some(&buf), 0, 0);
        surf.commit();
        h.roundtrip();
        let sid = h.server_surface_id(&surf);
        let mut got_initial = false;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let hdr = decode_rb(rb, &mut canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                assert!(!hdr.incremental);
                got_initial = true;
            }
        }
        assert!(got_initial);
        assert_eq!(canvas, frame);
        assert!(!h.state.synced_buffers[&sid].shadow.is_compressed());

        // 2. Sweep before timeout is a no-op; sweep at timeout compresses the dormant shadow.
        let t_active = h.state.synced_buffers[&sid].last_active;
        assert_eq!(
            h.state.compress_dormant_shadows(
                t_active + constants::DORMANT_SHADOW_IDLE_TIMEOUT - Duration::from_secs(1)
            ),
            0
        );
        assert!(!h.state.synced_buffers[&sid].shadow.is_compressed());

        assert_eq!(
            h.state
                .compress_dormant_shadows(t_active + constants::DORMANT_SHADOW_IDLE_TIMEOUT),
            1
        );
        let compressed_len = h.state.synced_buffers[&sid]
            .shadow
            .compressed_bytes()
            .expect("expected shadow to be compressed after idle timeout");
        assert!(compressed_len < byte_len / 10);

        // 3. Unchanged repaint on a dormant compressed surface skips both wire bytes and decompression.
        surf.attach(Some(&buf), 0, 0);
        surf.commit();
        h.roundtrip();
        let raw_bufs = h
            .drain_messages()
            .into_iter()
            .filter(|m| matches!(m, SendType::RawBuffer(_)))
            .count();
        assert_eq!(raw_bufs, 0);
        assert!(
            h.state.synced_buffers[&sid].shadow.is_compressed(),
            "unchanged repaint must not decompress a dormant shadow with valid quad hashes"
        );

        // 4. Dirty edit wakes the dormant surface via incremental diff (zero full-frame resync).
        for y in 20..36 {
            for x in 20..36 {
                let off = y * stride + x * 4;
                frame[off] ^= 0x55;
                frame[off + 1] ^= 0xAA;
            }
        }
        file.rewind().unwrap();
        file.write_all(&frame).unwrap();
        surf.attach(Some(&buf), 0, 0);
        surf.commit();
        h.roundtrip();
        let mut woke_hdr = None;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let hdr = decode_rb(rb, &mut canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                woke_hdr = Some(hdr);
            }
        }
        let woke_hdr = woke_hdr.expect("expected incremental RawBuffer on wake");
        assert!(woke_hdr.incremental);
        assert_eq!(woke_hdr.tile_rects.len(), 1);
        assert_eq!(canvas, frame);
        assert!(!h.state.synced_buffers[&sid].shadow.is_compressed());

        // 5. Re-compress and resize: full frame replaces compressed shadow cleanly without decompressing.
        let t_woke = h.state.synced_buffers[&sid].last_active;
        assert_eq!(
            h.state
                .compress_dormant_shadows(t_woke + constants::DORMANT_SHADOW_IDLE_TIMEOUT),
            1
        );
        assert!(h.state.synced_buffers[&sid].shadow.is_compressed());

        let new_w = 256usize;
        let new_h = 256usize;
        let new_stride = new_w * 4;
        let new_len = new_h * new_stride;
        let buf_resized = pool.create_buffer(
            0,
            new_w as i32,
            new_h as i32,
            new_stride as i32,
            wl_shm::Format::Xrgb8888,
            &h.qh,
            (),
        );
        let mut resized_canvas = vec![0u8; new_len];
        surf.attach(Some(&buf_resized), 0, 0);
        surf.commit();
        h.roundtrip();
        let mut resized_hdr = None;
        for msg in h.drain_messages() {
            if let SendType::RawBuffer(rb) = msg {
                let hdr = decode_rb(rb, &mut resized_canvas, &mut client_cache);
                h.ack_frame(hdr.frame_seq);
                resized_hdr = Some(hdr);
            }
        }
        let resized_hdr = resized_hdr.expect("expected full RawBuffer on resize");
        assert!(!resized_hdr.incremental);
        assert_eq!(resized_canvas, frame[..new_len]);
        assert!(!h.state.synced_buffers[&sid].shadow.is_compressed());
        assert_eq!(h.state.synced_buffers[&sid].shadow.len(), new_len);
    }
}
