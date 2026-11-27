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

/// Handlers for events from the wprs client.
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::mem;
use std::os::fd::OwnedFd;
use std::thread;

use nix::fcntl::OFlag;
use nix::unistd;
use smithay::backend::input::ButtonState;
use smithay::backend::input::KeyState as SmithayKeyState;
use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
use smithay::input::keyboard::FilterResult;
use smithay::input::keyboard::KeysymHandle;
use smithay::input::keyboard::ModifiersState;
use smithay::input::pointer::ButtonEvent;
use smithay::input::pointer::CursorImageStatus as SmithayCursorImageStatus;
use smithay::input::pointer::CursorImageSurfaceData;
use smithay::input::pointer::MotionEvent;
use smithay::output::Output;
use smithay::output::PhysicalProperties;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Rectangle;
use smithay::utils::SERIAL_COUNTER;
use smithay::utils::Serial;
use smithay::wayland::compositor;
use smithay::wayland::compositor::BufferAssignment as SmithayBufferAssignment;
use smithay::wayland::compositor::SubsurfaceCachedState;
use smithay::wayland::compositor::SurfaceAttributes;
use smithay::wayland::compositor::SurfaceData;
use smithay::wayland::compositor::TraversalAction;
use smithay::wayland::selection::primary_selection;
use smithay::wayland::shell::xdg::ToplevelStateSet;

use crate::args;
use crate::compositor_utils;
use crate::filtering;
use crate::prelude::*;
use crate::serialization::Capabilities;
use crate::serialization::CompressionMode;
use crate::serialization::Event;
use crate::serialization::RecvType;
use crate::serialization::Request;
use crate::serialization::SendType;
use crate::serialization::Serializer;
use crate::serialization::wayland::BufferAssignment;
use crate::serialization::wayland::ClientSurface;
use crate::serialization::wayland::CursorImage;
use crate::serialization::wayland::CursorImageStatus;
use crate::serialization::wayland::DataDestinationEvent;
use crate::serialization::wayland::DataDestinationRequest;
use crate::serialization::wayland::DataEvent;
use crate::serialization::wayland::DataRequest;
use crate::serialization::wayland::DataSource;
use crate::serialization::wayland::DataSourceEvent;
use crate::serialization::wayland::DataToTransfer;
use crate::serialization::wayland::DataTransferTarget;
use crate::serialization::wayland::KeyInner;
use crate::serialization::wayland::KeyState;
use crate::serialization::wayland::KeyboardEvent;
use crate::serialization::wayland::OutputEvent;
use crate::serialization::wayland::PointerEvent;
use crate::serialization::wayland::PointerEventKind;
use crate::serialization::wayland::RawModifiers;
use crate::serialization::wayland::RepeatInfo;
use crate::serialization::wayland::Role;
use crate::serialization::wayland::SubsurfacePosition;
use crate::serialization::wayland::SurfaceEvent;
use crate::serialization::wayland::SurfaceEventPayload;
use crate::serialization::wayland::SurfaceRequest;
use crate::serialization::wayland::SurfaceRequestPayload;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::WlSurfaceId;
use crate::serialization::xdg_shell::FullscreenState;
use crate::serialization::xdg_shell::PopupConfigure;
use crate::serialization::xdg_shell::PopupConfigureKind;
use crate::serialization::xdg_shell::PopupEvent;
use crate::serialization::xdg_shell::ToplevelConfigure;
use crate::serialization::xdg_shell::ToplevelEvent;
use crate::server::LockedSurfaceState;
use crate::server::SyncedSurfaceBuffer;
use crate::server::WprsServerState;
use crate::server::smithay_handlers::get_surface_z_order;
use crate::server::smithay_handlers::sync_subsurface_role;
use crate::sharding_compression::ShardingCompressor;

enum UnknownSurfaceErr {
    ObjectId(WlSurfaceId),
    Client(ObjectId),
    Surface(Client),
}

fn committed_wl_buffer(surface_data: &SurfaceData) -> Option<WlBuffer> {
    if let Some(SmithayBufferAssignment::NewBuffer(buffer)) = &surface_data
        .cached_state
        .get::<SurfaceAttributes>()
        .current()
        .buffer
    {
        return Some(buffer.clone());
    }

    surface_data
        .data_map
        .get::<RendererSurfaceStateUserData>()
        .and_then(|state| {
            state
                .lock()
                .unwrap()
                .buffer()
                .map(|buffer| WlBuffer::clone(buffer))
        })
}

fn sync_surface_data(
    serializer: &Serializer<Request, Event>,
    compressor: &mut ShardingCompressor,
    compression_mode: CompressionMode,
    synced_buffers: &mut HashMap<WlSurfaceId, SyncedSurfaceBuffer>,
    committed_surfaces: &mut HashSet<WlSurfaceId>,
    surface_data: &SurfaceData,
    parent_id: Option<WlSurfaceId>,
    sync: bool,
    z_ordered_children: Vec<SubsurfacePosition>,
) {
    // Uncommitted child subsurfaces visited by `with_surface_tree_downward` do
    // not have `LockedSurfaceState` yet and will sync on their first commit.
    let Some(locked_state) = surface_data.data_map.get::<LockedSurfaceState>() else {
        return;
    };
    let mut tracked = locked_state.0.lock().unwrap();
    if !tracked.has_committed {
        return;
    }
    let mut surface_state_to_send = tracked.state.clone();
    sync_subsurface_role(
        &mut surface_state_to_send,
        matches!(surface_data.role, Some("subsurface")),
        parent_id,
    );
    if let Some(Role::SubSurface(subsurface_state)) = &mut surface_state_to_send.role {
        subsurface_state.sync = sync;
        subsurface_state.location = surface_data
            .cached_state
            .get::<SubsurfaceCachedState>()
            .pending()
            .location
            .into();
    }
    if let Some(Role::XdgPopup(popup_state)) = &mut surface_state_to_send.role {
        // Never replay an explicit grab serial from a prior connection on sync.
        popup_state.grab_serial = None;
    }
    surface_state_to_send.z_ordered_children = z_ordered_children;
    // `wprsc` treats `damage: None` + `BufferAssignment::New` as full damage
    // (`FrameDamage::merge_commit`).
    surface_state_to_send.damage = None;

    if matches!(surface_state_to_send.buffer, Some(BufferAssignment::New(_))) {
        // Remove any prior shadow buffer before reading `wl_shm` so a `BadMap`
        // or invalid-metadata error cannot leave a stale shadow behind.
        let mut shadow_pixels = synced_buffers
            .remove(&surface_state_to_send.id)
            .map_or_else(Vec::new, |old| old.pixels);
        match committed_wl_buffer(surface_data) {
            Some(wl_buffer) => {
                let mut raw_buffer_sent = false;
                let writer = serializer.writer();
                let sync_res = compositor_utils::with_buffer_contents(
                    &wl_buffer,
                    |data, spec| -> Result<SyncedSurfaceBuffer> {
                        let metadata = surface_state_to_send.set_buffer(&spec).location(loc!())?;
                        compositor_utils::fault_probe_buffer(data, metadata.stride as usize, None);
                        filtering::filter_and_compress_full_streaming(
                            data,
                            surface_state_to_send.client,
                            surface_state_to_send.id,
                            metadata,
                            compression_mode,
                            Some(&mut shadow_pixels),
                            compressor,
                            |raw_buffer_to_send| {
                                raw_buffer_sent = true;
                                writer.send(SendType::RawBuffer(raw_buffer_to_send));
                            },
                        );
                        Ok(SyncedSurfaceBuffer {
                            metadata,
                            pixels: shadow_pixels,
                        })
                    },
                );
                match sync_res {
                    Ok(Ok(synced)) => {
                        synced_buffers.insert(surface_state_to_send.id, synced);
                    },
                    Ok(Err(err)) => {
                        warn!("skipping invalid buffer during surface sync: {err:?}");
                        surface_state_to_send.buffer = None;
                    },
                    Err(err) => {
                        // Smithay reports SIGBUS as `Err(BadMap)` after the
                        // closure returns, so `raw_buffer_sent` may already be
                        // `true`.
                        warn!("skipping unreadable shm buffer during surface sync: {err:?}");
                        surface_state_to_send.buffer =
                            raw_buffer_sent.then_some(BufferAssignment::Removed);
                    },
                }
            },
            None => {
                warn!(
                    "surface {:?} had BufferAssignment::New without committed WlBuffer; clearing buffer for sync",
                    surface_state_to_send.id
                );
                surface_state_to_send.buffer = None;
            },
        }
    }

    let client = surface_state_to_send.client;
    let surface = surface_state_to_send.id;
    serializer
        .writer()
        .send(SendType::Object(Request::Surface(SurfaceRequest {
            client,
            surface,
            payload: SurfaceRequestPayload::Commit(surface_state_to_send),
        })));
    committed_surfaces.insert(surface);
    if tracked.state.buffer.is_some() {
        tracked.dirty = false;
    }
}

impl WprsServerState {
    fn sync_surface_tree(&mut self, root: &WlSurface) {
        // Collect the tree first so `get_parent` and `get_surface_z_order`
        // never re-lock `PrivateSurfaceData` while `with_surface_tree_downward`
        // holds ancestor mutexes.
        let mut surfaces = Vec::new();
        compositor::with_surface_tree_downward(
            root,
            (),
            |_, _, _| TraversalAction::DoChildren(()),
            |surface, _, _| surfaces.push(surface.clone()),
            |_, _, _| true,
        );
        for surface in &surfaces {
            let parent_id = compositor::get_parent(surface)
                .as_ref()
                .map(WlSurfaceId::new);
            let sync = parent_id.is_some() && compositor::is_sync_subsurface(surface);
            let z_ordered_children = get_surface_z_order(surface);
            compositor::with_states(surface, |surface_data| {
                sync_surface_data(
                    &self.serializer,
                    &mut self.compressor,
                    self.compression_mode,
                    &mut self.synced_buffers,
                    &mut self.committed_surfaces,
                    surface_data,
                    parent_id,
                    sync,
                    z_ordered_children,
                );
            });
        }
    }

    pub(crate) fn sync_surface_if_unsynced(&mut self, surface: &WlSurface) {
        if self.serializer.other_end_connected()
            && !self.synced_buffers.contains_key(&WlSurfaceId::new(surface))
        {
            self.sync_surface_tree(surface);
        }
    }

    pub(crate) fn send_cursor_status(&mut self, image: &SmithayCursorImageStatus) {
        let cursor_image_status = match image {
            SmithayCursorImageStatus::Hidden => CursorImageStatus::Hidden,
            SmithayCursorImageStatus::Surface(surface) => {
                let client_surface = log_and_return!(ClientSurface::new(surface));
                log_and_return!(self.insert_surface(surface));

                let hotspot = compositor::with_states(surface, |surface_data| {
                    let hotspot = surface_data
                        .data_map
                        .get::<CursorImageSurfaceData>()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .hotspot;

                    let mut tracked = surface_data
                        .data_map
                        .get::<LockedSurfaceState>()
                        .unwrap()
                        .0
                        .lock()
                        .unwrap();
                    tracked.state.role = Some(Role::Cursor(hotspot.into()));
                    tracked.dirty = true;

                    hotspot
                });

                self.sync_surface_if_unsynced(surface);

                CursorImageStatus::Surface {
                    client_surface,
                    hotspot: hotspot.into(),
                }
            },
            SmithayCursorImageStatus::Named(name) => {
                CursorImageStatus::Named(name.name().to_string())
            },
        };

        // TODO: expose serial to this function, then remove last_enter_serial
        // on client.
        self.serializer
            .writer()
            .send(SendType::Object(Request::CursorImage(CursorImage {
                serial: 0,
                status: cursor_image_status,
            })));
    }
    fn object_client_surface_from_id(
        &self,
        surface_id: &WlSurfaceId,
    ) -> Result<(ObjectId, Client, WlSurface), UnknownSurfaceErr> {
        let object_id = match self.object_map.get(surface_id) {
            Some(object_id) => object_id.clone(),
            None => {
                return Err(UnknownSurfaceErr::ObjectId(*surface_id));
            },
        };
        let Ok(client) = self.dh.get_client(object_id.clone()) else {
            return Err(UnknownSurfaceErr::Client(object_id));
        };

        let Ok(surface) = client.object_from_protocol_id(&self.dh, object_id.protocol_id()) else {
            return Err(UnknownSurfaceErr::Surface(client));
        };

        Ok((object_id, client, surface))
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_pointer_frame(&mut self, events: Vec<PointerEvent>) -> Result<()> {
        let pointer = self.seat.get_pointer().location(loc!())?;

        for event in events {
            let res = self
                .object_client_surface_from_id(&event.surface_id)
                .map_err(|err| match err {
                    UnknownSurfaceErr::ObjectId(surface_id) => {
                        anyhow!("Ignoring pointer event for unknown object {surface_id:?}")
                    },
                    UnknownSurfaceErr::Client(object_id) => {
                        anyhow!("Ignoring pointer event for unknown client {object_id:?}")
                    },
                    UnknownSurfaceErr::Surface(client) => {
                        anyhow!("Ignoring pointer event for unknown surface {client:?}")
                    },
                })
                .warn(loc!());

            let surface = match res {
                Ok((_, _, surface)) => surface,
                Err(_) => {
                    // We do not want to propogate this error since we already warned about it.
                    continue;
                },
            };

            let time = self.start_time.elapsed().as_millis() as u32;

            match event.kind {
                PointerEventKind::Enter { serial } => {
                    debug!("pointer entered at {:?}", event.position);
                    let serial = self.serial_map.insert(serial);
                    pointer.motion(
                        self,
                        Some((surface, (0 as f64, 0 as f64).into())),
                        &MotionEvent {
                            location: event.position.into(),
                            serial,
                            time,
                        },
                    );
                },
                PointerEventKind::Leave { serial } => {
                    debug!("pointer left");

                    // During a proxied drag/drop, the host compositor sends
                    // `wl_pointer.leave` and drives the session via
                    // `wl_data_device` events instead, so keep pressed buttons
                    // until the drag grab ends.
                    if !self.data_device_state.is_proxy_dnd_grab_active() {
                        let pressed_buttons: HashSet<u32> = mem::take(&mut self.pressed_buttons);
                        for button in pressed_buttons {
                            debug!("releasing button {}", button);
                            pointer.button(
                                self,
                                &ButtonEvent {
                                    time,
                                    button,
                                    serial: SERIAL_COUNTER.next_serial(),
                                    state: ButtonState::Released,
                                },
                            );
                        }
                    }

                    let serial = self.serial_map.insert(serial);
                    pointer.motion(
                        self,
                        None,
                        &MotionEvent {
                            location: event.position.into(),
                            serial,
                            time,
                        },
                    );
                },
                PointerEventKind::Motion => {
                    debug!("pointer moved to {:?}", event.position);
                    pointer.motion(
                        self,
                        Some((surface, (0 as f64, 0 as f64).into())),
                        &MotionEvent {
                            location: event.position.into(),
                            serial: 0.into(), // unused
                            time,
                        },
                    );
                },
                PointerEventKind::Press { serial, button } => {
                    debug!("button {:x} pressed at {:?}", button, event.position);
                    let serial = self.serial_map.insert(serial);
                    pointer.button(
                        self,
                        &ButtonEvent {
                            time,
                            button,
                            serial,
                            state: ButtonState::Pressed,
                        },
                    );
                    self.pressed_buttons.insert(button);
                },
                PointerEventKind::Release { serial, button } => {
                    debug!("button {:x} released at {:?}", button, event.position);
                    let serial = self.serial_map.insert(serial);
                    pointer.button(
                        self,
                        &ButtonEvent {
                            time,
                            button,
                            serial,
                            state: ButtonState::Released,
                        },
                    );
                    self.pressed_buttons.remove(&button);
                },
                PointerEventKind::Axis {
                    horizontal,
                    vertical,
                    source,
                } => {
                    debug!(
                        "axis event: horizontal {horizontal:?}, vertical {vertical:?}, source {source:?}"
                    );
                    pointer.axis(
                        self,
                        compositor_utils::build_axis_frame(time, horizontal, vertical, source),
                    );
                },
            }
        }
        pointer.frame(self);

        Ok(())
    }

    #[instrument(
        skip(self, keycode, state),
        fields(keycode = "<redacted>", state = "<redacted>"),
        level = "debug"
    )]
    fn set_key_state(&mut self, keycode: u32, state: KeyState, serial: Serial) -> Result<()> {
        let keyboard = self.seat.get_keyboard().location(loc!())?;

        if args::get_log_priv_data() {
            debug!("sending key input: code {keycode:?}, state {state:?}");
        }

        fn filter(
            _: &mut WprsServerState,
            modifiers_state: &ModifiersState,
            keysym: KeysymHandle,
        ) -> FilterResult<()> {
            if args::get_log_priv_data() {
                debug!("modifiers_state {modifiers_state:?}, keysym {keysym:?}");
            }
            FilterResult::Forward
        }

        // our keycode is getting offset by 8 for reasons
        // see https://github.com/Smithay/smithay/pull/1536
        let x11_keycode = keycode
            .checked_add(8)
            .context(loc!(), "keycode overflow")?
            .into();
        let time = self.start_time.elapsed().as_millis() as u32;
        match state {
            KeyState::Pressed => {
                if !self.pressed_keys.insert(keycode) {
                    return Ok(());
                }
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Pressed,
                    serial,
                    time,
                    filter,
                );
            },
            KeyState::Released => {
                if !self.pressed_keys.remove(&keycode) {
                    return Ok(());
                }
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Released,
                    serial,
                    time,
                    filter,
                );
            },
            KeyState::Repeated => {
                // Map repeated to released + pressed
                // Smithay 0.7 keystates don't support repetition
                if self.pressed_keys.contains(&keycode) {
                    keyboard.input::<(), _>(
                        self,
                        x11_keycode,
                        SmithayKeyState::Released,
                        serial,
                        time,
                        filter,
                    );
                }
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Pressed,
                    serial,
                    time,
                    filter,
                );
                self.pressed_keys.insert(keycode);
            },
        }

        Ok(())
    }

    fn release_all_pressed_keys(&mut self) -> Result<()> {
        let pressed: Vec<u32> = self.pressed_keys.iter().copied().collect();
        for keycode in pressed {
            self.set_key_state(keycode, KeyState::Released, SERIAL_COUNTER.next_serial())
                .location(loc!())?;
        }
        Ok(())
    }

    fn reset_input_state(&mut self) {
        if let Some(keyboard) = self.seat.get_keyboard() {
            keyboard.set_focus(self, None, SERIAL_COUNTER.next_serial());
        }
        self.data_device_state.set_clipboard_focus(&self.dh, None);
        primary_selection::set_primary_focus(&self.dh, &self.seat, None);
        self.release_all_pressed_keys().log_and_ignore(loc!());
        let seat = self.seat.clone();
        compositor_utils::sync_keyboard_modifiers(
            &seat,
            self,
            RawModifiers::default(),
            0,
            SERIAL_COUNTER.next_serial(),
        );

        if let Some(pointer) = self.seat.get_pointer() {
            let time = self.start_time.elapsed().as_millis() as u32;
            let pressed_buttons: HashSet<u32> = mem::take(&mut self.pressed_buttons);
            for button in pressed_buttons {
                pointer.button(
                    self,
                    &ButtonEvent {
                        time,
                        button,
                        serial: SERIAL_COUNTER.next_serial(),
                        state: ButtonState::Released,
                    },
                );
            }
            pointer.motion(
                self,
                None,
                &MotionEvent {
                    location: (0.0, 0.0).into(),
                    serial: SERIAL_COUNTER.next_serial(),
                    time,
                },
            );
            pointer.frame(self);
        } else {
            self.pressed_buttons.clear();
        }
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_keyboard_event(&mut self, event: KeyboardEvent) -> Result<()> {
        let keyboard = self.seat.get_keyboard().location(loc!())?;
        match event {
            KeyboardEvent::Enter {
                serial,
                surface_id,
                keycodes,
                keysyms: _,
            } => {
                // Validate the target surface before mutating focus or pressed keys so
                // stale/unknown `Enter` events never leak stuck keys into Smithay.
                let (_, client, surface) = self
                    .object_client_surface_from_id(&surface_id)
                    .map_err(|err| match err {
                        UnknownSurfaceErr::ObjectId(surface_id) => {
                            anyhow!("Ignoring keyboard event for unknown object {surface_id:?}")
                        },
                        UnknownSurfaceErr::Client(object_id) => {
                            anyhow!("Ignoring keyboard event for unknown client {object_id:?}")
                        },
                        UnknownSurfaceErr::Surface(client) => {
                            anyhow!("Ignoring keyboard event for unknown surface {client:?}")
                        },
                    })
                    .warn(loc!())?;

                // Clear any stale focus and pressed keys (e.g. if a previous focused
                // surface was destroyed without a host `Leave`) before simulating the
                // entered surface's held keys while unfocused.
                if keyboard.is_focused() {
                    keyboard.set_focus(self, None, SERIAL_COUNTER.next_serial());
                }
                self.release_all_pressed_keys().location(loc!())?;

                // We simulate keycodes before focusing since that is what a normal wayland application would see.
                // Process modifier keys first so that they apply to other held keys.
                let mut delayed_keycodes = Vec::new();
                for keycode in keycodes {
                    if compositor_utils::is_modifier_keycode(keycode) {
                        self.set_key_state(
                            keycode,
                            KeyState::Pressed,
                            SERIAL_COUNTER.next_serial(),
                        )
                        .location(loc!())?;
                    } else {
                        delayed_keycodes.push(keycode);
                    }
                }
                for keycode in delayed_keycodes {
                    self.set_key_state(keycode, KeyState::Pressed, SERIAL_COUNTER.next_serial())
                        .location(loc!())?;
                }

                let serial = self.serial_map.insert(serial);
                debug!("setting keyboard focus to surface {surface:?}");
                keyboard.set_focus(self, Some(surface), serial);
                self.data_device_state
                    .set_clipboard_focus(&self.dh, Some(client.clone()));
                primary_selection::set_primary_focus(&self.dh, &self.seat, Some(client));
            },
            KeyboardEvent::Leave { serial } => {
                let serial = self.serial_map.insert(serial);
                keyboard.set_focus(self, None, serial);
                self.data_device_state.set_clipboard_focus(&self.dh, None);
                primary_selection::set_primary_focus(&self.dh, &self.seat, None);
                self.release_all_pressed_keys().location(loc!())?;
            },
            KeyboardEvent::Key(KeyInner {
                serial,
                raw_code,
                state: istate,
            }) => {
                let serial = self.serial_map.insert(serial);

                self.set_key_state(raw_code, istate, serial)
                    .location(loc!())?;
            },
            KeyboardEvent::RepeatInfo(info) => match info {
                RepeatInfo::Repeat { rate, delay } => {
                    keyboard.change_repeat_info(
                        i32::try_from(u32::from(rate)).location(loc!())?,
                        i32::try_from(delay).location(loc!())?,
                    );
                },
                RepeatInfo::Disable => {
                    keyboard.change_repeat_info(0, 0);
                },
            },
            KeyboardEvent::Keymap(keymap) => keyboard
                .set_keymap_from_string(self, keymap)
                .location(loc!())?,
            KeyboardEvent::Modifiers {
                serial,
                raw_modifiers,
                layout_index,
            } => {
                let serial = self.serial_map.insert(serial);
                let seat = self.seat.clone();
                compositor_utils::sync_keyboard_modifiers(
                    &seat,
                    self,
                    raw_modifiers,
                    layout_index,
                    serial,
                );
            },
        }

        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_toplevel_configure(&self, configure: &ToplevelConfigure) -> Result<()> {
        let surfaces = self.xdg_shell_state.toplevel_surfaces();
        // TODO: we can replace this with a hashmap lookup now
        surfaces
            .iter()
            .find(|surface| {
                let surface_id = WlSurfaceId::new(surface.wl_surface());
                debug!(
                    "inspecting surface {surface_id:?}, looking for surface {:?}",
                    configure.surface_id
                );
                surface_id == configure.surface_id
            })
            .map(|surface| {
                let surface_id = WlSurfaceId::new(surface.wl_surface());
                debug!("matched surface {surface_id:?}");
                let size = Some(
                    (
                        configure.new_size.w.map_or(0i32, |w| u32::from(w) as i32),
                        configure.new_size.h.map_or(0i32, |h| u32::from(h) as i32),
                    )
                        .into(),
                );
                let states: ToplevelStateSet = configure.state.into();
                let fullscreen_output = if states.contains(xdg_toplevel::State::Fullscreen) {
                    let requested_output_id =
                        compositor::with_states(surface.wl_surface(), |surface_data| {
                            surface_data
                                .data_map
                                .get::<LockedSurfaceState>()
                                .and_then(|locked| match &locked.0.lock().unwrap().state.role {
                                    Some(Role::XdgToplevel(tl)) => match tl.fullscreen {
                                        Some(FullscreenState::Set(output_id)) => output_id,
                                        _ => None,
                                    },
                                    _ => None,
                                })
                        });
                    requested_output_id
                        .and_then(|id| self.client_wl_output_for_id(surface.wl_surface(), id))
                } else {
                    None
                };
                surface.with_pending_state(|ref mut state| {
                    state.size = size;
                    state.states = states;
                    state.fullscreen_output = fullscreen_output;
                    state.decoration_mode = Some(configure.decoration_mode.into());
                });
                surface.send_configure();
                debug!("sent configure to surface {surface:?}");
            });

        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_toplevel(&mut self, toplevel: ToplevelEvent) -> Result<()> {
        match &toplevel {
            ToplevelEvent::Configure(configure) => {
                self.handle_toplevel_configure(configure).location(loc!())?;
            },
            ToplevelEvent::Close(close) => {
                let surfaces = self.xdg_shell_state.toplevel_surfaces();
                if let Some(surface) = surfaces.iter().find(|surface| {
                    let surface_id = WlSurfaceId::new(surface.wl_surface());

                    surface_id == close.surface_id
                }) {
                    surface.send_close();
                }
            },
        }
        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_popup_configure(&self, configure: &PopupConfigure) -> Result<()> {
        let surfaces = self.xdg_shell_state.popup_surfaces();
        surfaces
            .iter()
            .find(|surface| {
                let surface_id = WlSurfaceId::new(surface.wl_surface());
                debug!(
                    "inspecting surface {surface_id:?}, looking for surface {:?}",
                    configure.surface_id
                );
                surface_id == configure.surface_id
            })
            .map(|surface| {
                let surface_id = WlSurfaceId::new(surface.wl_surface());
                debug!("matched surface {surface_id:?}");
                surface.with_pending_state(|state| {
                    state.geometry = Rectangle::new(
                        configure.position.into(),
                        (configure.width, configure.height).into(),
                    );
                });
                match configure.kind {
                    PopupConfigureKind::Reposition { token } => {
                        surface.send_repositioned(token);
                    },
                    PopupConfigureKind::Initial | PopupConfigureKind::Reactive => {
                        surface.send_configure().log_and_ignore(loc!());
                    },
                }
            })
            .location(loc!())?;

        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_popup(&mut self, popup: PopupEvent) -> Result<()> {
        match &popup {
            PopupEvent::Configure(configure) => {
                self.handle_popup_configure(configure).location(loc!())?;
            },
            PopupEvent::Done(surface_id) => {
                if let Some(surface) = self
                    .xdg_shell_state
                    .popup_surfaces()
                    .iter()
                    .find(|surface| WlSurfaceId::new(surface.wl_surface()) == *surface_id)
                {
                    surface.send_popup_done();
                }
            },
        }
        Ok(())
    }

    fn for_each_mapped_surface(&self, mut f: impl FnMut(&WlSurface, &mut SurfaceState)) {
        for surface_id in self.object_map.keys() {
            let Ok((_, _, surface)) = self.object_client_surface_from_id(surface_id) else {
                continue;
            };
            compositor::with_states(&surface, |surface_data| {
                let mut tracked = surface_data
                    .data_map
                    .get::<LockedSurfaceState>()
                    .expect("surfaces in object_map have LockedSurfaceState")
                    .0
                    .lock()
                    .unwrap();
                f(&surface, &mut tracked.state);
            });
        }
    }

    fn remove_output_from_surfaces(&self, output_id: u32, output: Option<&Output>) {
        self.for_each_mapped_surface(|surface, surface_state| {
            if let Some(output) = output {
                output.leave(surface);
            }
            surface_state.output_ids.retain(|id| *id != output_id);
            if let Some(Role::XdgToplevel(tl)) = &mut surface_state.role
                && tl.fullscreen == Some(FullscreenState::Set(Some(output_id)))
            {
                tl.fullscreen = Some(FullscreenState::Set(None));
            }
        });
    }

    fn enter_surfaces_on_output(&self, output_id: u32, output: &Output) {
        self.for_each_mapped_surface(|surface, surface_state| {
            if surface_state.output_ids.contains(&output_id) {
                output.enter(surface);
            }
        });
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_output(&mut self, output_event: OutputEvent) -> Result<()> {
        match output_event {
            OutputEvent::New(output) => {
                let output_id = output.id;
                let local_output = match self.outputs.entry(output_id) {
                    Entry::Occupied(mut entry) => {
                        let local_output = &mut entry.get_mut().0;
                        compositor_utils::update_output(local_output, output);
                        local_output.clone()
                    },
                    Entry::Vacant(entry) => {
                        let mut new_output = Output::new(
                            format!("{}_{}", output_id, output.name.as_deref().unwrap_or("None")),
                            PhysicalProperties {
                                size: output.physical_size.into(),
                                subpixel: output.subpixel.into(),
                                make: output.make.clone(),
                                model: output.model.clone(),
                            },
                        );
                        compositor_utils::update_output(&mut new_output, output);
                        let global_id = new_output.create_global::<Self>(&self.dh);
                        entry.insert((new_output.clone(), global_id));
                        new_output
                    },
                };
                // Host compositors with `zxdg_output_v1` can emit `wl_surface.enter`
                // (`OutputsChanged`) before `zxdg_output_v1.done` unblocks `OutputEvent::New`.
                self.enter_surfaces_on_output(output_id, &local_output);
            },
            OutputEvent::Update(output) => {
                let (local_output, _) = match self.outputs.entry(output.id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(_) => {
                        warn!("update to unknown display {:?}", output.id);
                        return Ok(());
                    },
                };

                compositor_utils::update_output(local_output, output);
            },
            OutputEvent::Destroy(output) => {
                let removed = self.outputs.remove(&output.id);
                // Always prune `output.id` even if `Destroy` arrives before
                // `zxdg_output_v1.done` unblocked `OutputEvent::New`.
                self.remove_output_from_surfaces(
                    output.id,
                    removed.as_ref().map(|(local_output, _)| local_output),
                );
                if let Some((_, global_id)) = removed {
                    self.dh.remove_global::<Self>(global_id);
                }
            },
        };

        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn handle_connect(&mut self) -> Result<()> {
        self.reset_dnd_state();
        self.reset_input_state();
        self.serial_map.clear();
        let old_outputs: Vec<_> = self.outputs.drain().collect();
        // Clear all previous-session `output_ids` unconditionally (including IDs
        // recorded by `OutputsChanged` before `OutputEvent::New` arrived), and
        // clear any pending popup grab serials captured in a previous session.
        self.for_each_mapped_surface(|surface, surface_state| {
            for (_, (local_output, _)) in &old_outputs {
                local_output.leave(surface);
            }
            surface_state.output_ids.clear();
            match &mut surface_state.role {
                Some(Role::XdgPopup(popup_state)) => {
                    popup_state.grab_serial = None;
                },
                Some(Role::XdgToplevel(toplevel_state)) => {
                    if let Some(FullscreenState::Set(output_id)) = &mut toplevel_state.fullscreen {
                        *output_id = None;
                    }
                },
                _ => {},
            }
        });
        for (_, (_, global_id)) in old_outputs {
            self.dh.remove_global::<Self>(global_id);
        }
        // Reset compression mode to Lossless on each new connection so a
        // lossless client never inherits a prior hybrid client's setting.
        self.compression_mode = CompressionMode::Lossless;
        self.synced_buffers.clear();
        self.dirty_sync_surfaces.clear();
        self.committed_surfaces.clear();

        self.serializer
            .writer()
            .send(SendType::Object(Request::Capabilities(Capabilities {
                xwayland: self.xwayland_enabled,
            })));

        let roots: Vec<WlSurface> = self
            .xdg_shell_state
            .toplevel_surfaces()
            .iter()
            .map(|s| s.wl_surface().clone())
            .chain(
                self.xdg_shell_state
                    .popup_surfaces()
                    .iter()
                    .map(|s| s.wl_surface().clone()),
            )
            .collect();
        for root in &roots {
            self.sync_surface_tree(root);
        }

        if let Some(cursor_status) = self.cursor_status.clone() {
            self.send_cursor_status(&cursor_status);
        }

        Ok(())
    }

    pub(crate) fn request_host_data_transfer(
        &mut self,
        target: DataTransferTarget,
        mime_type: String,
        fd: OwnedFd,
    ) {
        if !self.serializer.other_end_connected() {
            return;
        }
        let transfer_id = self.next_transfer_id;
        self.next_transfer_id += 1;
        self.pending_receive_pipes.insert(transfer_id, fd);
        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(
                DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer {
                    target,
                    mime_type,
                    transfer_id,
                }),
            )));
    }

    #[allow(clippy::verbose_file_reads)]
    #[instrument(skip_all, level = "debug")]
    fn handle_data_event(&mut self, data_event: DataEvent) -> Result<()> {
        match data_event {
            DataEvent::SourceEvent(DataSourceEvent::DnDMimeTypeAcceptedByDestination {
                source_id,
                mime_type,
            }) => {
                self.handle_dnd_source_target(source_id, mime_type);
            },
            DataEvent::SourceEvent(DataSourceEvent::MimeTypeSendRequestedByDestination {
                target,
                mime_type,
                transfer_id,
            }) => {
                let writer = self.serializer.writer();
                let pipe_res = unistd::pipe2(OFlag::O_CLOEXEC).location(loc!());
                let (recv_fd, send_fd) = match pipe_res {
                    Ok(fds) => fds,
                    Err(err) => {
                        writer.send(SendType::Object(Request::Data(DataRequest::TransferData(
                            transfer_id,
                            DataToTransfer(Vec::new()),
                        ))));
                        return Err(err);
                    },
                };

                let send_res = match target {
                    DataTransferTarget::Selection => self
                        .data_device_state
                        .request_client_clipboard(mime_type, send_fd),
                    DataTransferTarget::Primary => {
                        primary_selection::request_primary_client_selection(
                            &self.seat, mime_type, send_fd,
                        )
                        .location(loc!())
                    },
                    DataTransferTarget::DnD(source_id) => self
                        .data_device_state
                        .request_client_dnd_send(source_id, mime_type, send_fd),
                };
                if let Err(err) = send_res {
                    writer.send(SendType::Object(Request::Data(DataRequest::TransferData(
                        transfer_id,
                        DataToTransfer(Vec::new()),
                    ))));
                    return Err(err);
                }

                let mut f = File::from(recv_fd);
                // The data source application will write to the other end of
                // the pipe at its convenience and then close the file
                // descriptor, so spawn a thread to read until EOF and forward
                // the bytes back to `wprsc`.
                thread::spawn(move || {
                    debug!("in receive read thread for transfer {transfer_id}");
                    let mut buf = Vec::new();
                    let bytes_read = f.read_to_end(&mut buf);
                    debug!("read selection ({bytes_read:?} bytes): {buf:?}");
                    writer.send(SendType::Object(Request::Data(DataRequest::TransferData(
                        transfer_id,
                        DataToTransfer(buf),
                    ))));
                });
            },
            DataEvent::SourceEvent(DataSourceEvent::DnDActionSelected { source_id, action }) => {
                self.handle_dnd_source_action(source_id, action)
                    .location(loc!())?;
            },
            DataEvent::SourceEvent(DataSourceEvent::DnDDropPerformed { source_id }) => {
                self.handle_dnd_source_drop_performed(source_id);
            },
            DataEvent::SourceEvent(DataSourceEvent::DnDFinished { source_id }) => {
                self.handle_dnd_source_finished_or_cancelled(source_id, true);
            },
            DataEvent::SourceEvent(DataSourceEvent::DnDCancelled { source_id }) => {
                self.handle_dnd_source_finished_or_cancelled(source_id, false);
            },
            DataEvent::DestinationEvent(DataDestinationEvent::DnDSourceActions {
                offer_id,
                actions,
            }) => {
                self.handle_dnd_offer_source_actions(offer_id, actions)
                    .location(loc!())?;
            },
            DataEvent::DestinationEvent(DataDestinationEvent::DnDActionSelected {
                offer_id,
                action,
            }) => {
                self.handle_dnd_offer_selected_action(offer_id, action)
                    .location(loc!())?;
            },
            DataEvent::DestinationEvent(DataDestinationEvent::DnDEnter(drag_enter)) => {
                let (_, _, surface) = self
                    .object_client_surface_from_id(&drag_enter.surface)
                    .map_err(|err| match err {
                        UnknownSurfaceErr::ObjectId(surface_id) => {
                            anyhow!("Ignoring DnDEnter event for unknown object {surface_id:?}")
                        },
                        UnknownSurfaceErr::Client(object_id) => {
                            anyhow!("Ignoring DnDEnter event for unknown client {object_id:?}")
                        },
                        UnknownSurfaceErr::Surface(client) => {
                            anyhow!("Ignoring DnDEnter event for unknown surface {client:?}")
                        },
                    })
                    .warn(loc!())?;
                self.handle_dnd_enter(drag_enter, surface)
                    .location(loc!())?;
            },
            DataEvent::DestinationEvent(DataDestinationEvent::DnDLeave { offer_id }) => {
                debug!("drag leave {offer_id}");
                self.handle_dnd_leave(offer_id);
            },
            DataEvent::DestinationEvent(DataDestinationEvent::DnDMotion { offer_id, loc }) => {
                debug!("drag moved to {loc:?} for offer {offer_id}");
                self.handle_dnd_motion(offer_id, loc.x, loc.y);
            },
            DataEvent::DestinationEvent(DataDestinationEvent::DnDDrop { offer_id }) => {
                debug!("drag dropped for offer {offer_id}");
                self.handle_dnd_drop(offer_id);
            },
            DataEvent::DestinationEvent(DataDestinationEvent::SelectionSet(source, metadata)) => {
                match source {
                    DataSource::Selection => self
                        .data_device_state
                        .set_host_clipboard_selection(&self.dh, metadata.mime_types),
                    DataSource::Primary => primary_selection::set_primary_selection(
                        &self.dh,
                        &self.seat,
                        metadata.mime_types,
                        (),
                    ),
                };
            },
            DataEvent::TransferData(transfer_id, data) => {
                let fd = self
                    .pending_receive_pipes
                    .remove(&transfer_id)
                    .location(loc!())?;
                if data.0.is_empty() {
                    return Ok(());
                }
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

    #[instrument(skip_all, level = "debug")]
    fn handle_surface_event(&mut self, surface_event: SurfaceEvent) -> Result<()> {
        let outputs = match surface_event.payload {
            SurfaceEventPayload::BufferResyncRequested => {
                // Clear `synced_buffers` unconditionally before resolving the
                // surface so the next application commit transmits a full frame
                // without triggering an eager retry loop if client `/dev/shm` is full.
                self.synced_buffers.remove(&surface_event.surface_id);
                return Ok(());
            },
            SurfaceEventPayload::OutputsChanged(outputs) => outputs,
        };

        let (_, _, surface) = self
            .object_client_surface_from_id(&surface_event.surface_id)
            .map_err(|err| match err {
                UnknownSurfaceErr::ObjectId(surface_id) => anyhow!(
                    "Ignoring OutputsChanged({outputs:?}) event for unknown object {surface_id:?}"
                ),
                UnknownSurfaceErr::Client(object_id) => anyhow!(
                    "Ignoring OutputsChanged({outputs:?}) event for unknown client {object_id:?}"
                ),
                UnknownSurfaceErr::Surface(client) => anyhow!(
                    "Ignoring OutputsChanged({outputs:?}) event for unknown surface {client:?}"
                ),
            })
            .warn(loc!())?;

        compositor::with_states(&surface, |surface_data| {
            let locked_state = surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .expect("surfaces in object_map have LockedSurfaceState");
            let surface_state = &mut locked_state.0.lock().unwrap().state;

            let new_ids: HashSet<u32> = HashSet::from_iter(outputs.iter().map(|output| output.id));
            let old_ids = HashSet::from_iter(surface_state.output_ids.iter().cloned());

            compositor_utils::update_surface_outputs(&surface, &new_ids, &old_ids, |id| {
                self.outputs.get(id).map(|(output, _)| output)
            });

            surface_state.output_ids = new_ids.iter().cloned().collect();
        });

        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn handle_set_compression_mode(&mut self, mode: CompressionMode) -> Result<()> {
        mode.validate().location(loc!())?;
        info!("client set compression mode: {mode:?}");
        self.compression_mode = mode;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    pub fn handle_event(&mut self, conn_id: u64, event: RecvType<Event>) {
        if matches!(event, RecvType::Connected) {
            if !self.serializer.activate_connection(conn_id) {
                debug!("ignoring Connected from already-disconnected session {conn_id}");
                return;
            }
        } else if !self.serializer.is_active_connection(conn_id) {
            debug!("ignoring stale event from session {conn_id}");
            return;
        }

        match event {
            RecvType::Connected => self.handle_connect(),
            RecvType::Object(Event::Toplevel(toplevel)) => self.handle_toplevel(toplevel),
            RecvType::Object(Event::Popup(popup)) => self.handle_popup(popup),
            RecvType::Object(Event::KeyboardEvent(event)) => self.handle_keyboard_event(event),
            RecvType::Object(Event::PointerFrame(events)) => self.handle_pointer_frame(events),
            RecvType::Object(Event::Output(output_event)) => self.handle_output(output_event),
            RecvType::Object(Event::Data(data_event)) => self.handle_data_event(data_event),
            RecvType::Object(Event::Surface(surface_event)) => {
                self.handle_surface_event(surface_event)
            },
            RecvType::Object(Event::SetCompressionMode(mode)) => {
                self.handle_set_compression_mode(mode)
            },
            RecvType::RawBuffer(_) => unreachable!(),
        }
        .log_and_ignore(loc!());
        // TODO: maybe send errors back to the client.
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches;
    use std::num::NonZeroU32;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use smithay::input::keyboard::xkb;
    use smithay::reexports::calloop::EventLoop;
    use smithay::reexports::wayland_server::Display;
    use smithay_client_toolkit::reexports::client::Connection;
    use smithay_client_toolkit::reexports::client::Dispatch as ClientDispatch;
    use smithay_client_toolkit::reexports::client::EventQueue;
    use smithay_client_toolkit::reexports::client::Proxy;
    use smithay_client_toolkit::reexports::client::QueueHandle;
    use smithay_client_toolkit::reexports::client::WEnum as ClientWEnum;
    use smithay_client_toolkit::reexports::client::protocol::wl_callback::WlCallback;
    use smithay_client_toolkit::reexports::client::protocol::wl_compositor::WlCompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_keyboard;
    use smithay_client_toolkit::reexports::client::protocol::wl_keyboard::WlKeyboard as ClientWlKeyboard;
    use smithay_client_toolkit::reexports::client::protocol::wl_pointer;
    use smithay_client_toolkit::reexports::client::protocol::wl_pointer::WlPointer as ClientWlPointer;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry::WlRegistry;
    use smithay_client_toolkit::reexports::client::protocol::wl_seat;
    use smithay_client_toolkit::reexports::client::protocol::wl_seat::WlSeat as ClientWlSeat;
    use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as ClientWlSurface;

    use super::*;
    use crate::serialization::JpegSubsamp;
    use crate::server::smithay_handlers::ClientState;

    #[derive(Debug, Clone, Eq, PartialEq)]
    enum KbdWireEvent {
        Enter {
            surface_id: u32,
            keys: Vec<u32>,
        },
        Leave {
            surface_id: u32,
        },
        Key {
            key: u32,
            pressed: bool,
        },
        Modifiers {
            depressed: u32,
            latched: u32,
            locked: u32,
            group: u32,
        },
        RepeatInfo {
            rate: i32,
            delay: i32,
        },
    }

    #[derive(Default)]
    struct TestWaylandApp {
        compositor: Option<WlCompositor>,
        seat: Option<ClientWlSeat>,
        keyboard: Option<ClientWlKeyboard>,
        pointer: Option<ClientWlPointer>,
        kbd_events: Vec<KbdWireEvent>,
        pointer_leaves: Vec<u32>,
        pointer_button_releases: Vec<u32>,
    }

    impl ClientDispatch<WlRegistry, ()> for TestWaylandApp {
        fn event(
            state: &mut Self,
            registry: &WlRegistry,
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
                    "wl_seat" => {
                        state.seat =
                            Some(registry.bind::<ClientWlSeat, _, _>(name, version.min(9), qh, ()));
                    },
                    _ => {},
                }
            }
        }
    }

    impl ClientDispatch<WlCompositor, ()> for TestWaylandApp {
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

    impl ClientDispatch<ClientWlSurface, ()> for TestWaylandApp {
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

    impl ClientDispatch<ClientWlSeat, ()> for TestWaylandApp {
        fn event(
            state: &mut Self,
            seat: &ClientWlSeat,
            event: wl_seat::Event,
            _data: &(),
            _conn: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_seat::Event::Capabilities {
                capabilities: ClientWEnum::Value(caps),
            } = event
            {
                if caps.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                    state.keyboard = Some(seat.get_keyboard(qh, ()));
                }
                if caps.contains(wl_seat::Capability::Pointer) && state.pointer.is_none() {
                    state.pointer = Some(seat.get_pointer(qh, ()));
                }
            }
        }
    }

    impl ClientDispatch<ClientWlKeyboard, ()> for TestWaylandApp {
        fn event(
            state: &mut Self,
            _kbd: &ClientWlKeyboard,
            event: wl_keyboard::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                wl_keyboard::Event::Enter { surface, keys, .. } => {
                    let keys = keys
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .copied()
                        .map(u32::from_ne_bytes)
                        .collect();
                    state.kbd_events.push(KbdWireEvent::Enter {
                        surface_id: surface.id().protocol_id(),
                        keys,
                    });
                },
                wl_keyboard::Event::Leave { surface, .. } => {
                    state.kbd_events.push(KbdWireEvent::Leave {
                        surface_id: surface.id().protocol_id(),
                    });
                },
                wl_keyboard::Event::Key {
                    key,
                    state: ClientWEnum::Value(key_state),
                    ..
                } => {
                    let pressed = match key_state {
                        wl_keyboard::KeyState::Pressed => true,
                        wl_keyboard::KeyState::Released => false,
                        _ => return,
                    };
                    state.kbd_events.push(KbdWireEvent::Key { key, pressed });
                },
                wl_keyboard::Event::Modifiers {
                    mods_depressed,
                    mods_latched,
                    mods_locked,
                    group,
                    ..
                } => {
                    state.kbd_events.push(KbdWireEvent::Modifiers {
                        depressed: mods_depressed,
                        latched: mods_latched,
                        locked: mods_locked,
                        group,
                    });
                },
                wl_keyboard::Event::RepeatInfo { rate, delay } => {
                    state
                        .kbd_events
                        .push(KbdWireEvent::RepeatInfo { rate, delay });
                },
                _ => {},
            }
        }
    }

    impl ClientDispatch<ClientWlPointer, ()> for TestWaylandApp {
        fn event(
            state: &mut Self,
            _ptr: &ClientWlPointer,
            event: wl_pointer::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                wl_pointer::Event::Leave { surface, .. } => {
                    state.pointer_leaves.push(surface.id().protocol_id());
                },
                wl_pointer::Event::Button {
                    button,
                    state: ClientWEnum::Value(wl_pointer::ButtonState::Released),
                    ..
                } => {
                    state.pointer_button_releases.push(button);
                },
                _ => {},
            }
        }
    }

    impl ClientDispatch<WlCallback, Arc<AtomicBool>> for TestWaylandApp {
        fn event(
            _state: &mut Self,
            _proxy: &WlCallback,
            _event: <WlCallback as Proxy>::Event,
            done: &Arc<AtomicBool>,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            done.store(true, Ordering::Release);
        }
    }

    fn roundtrip_app(
        display: &mut Display<WprsServerState>,
        server: &mut WprsServerState,
        conn: &Connection,
        queue: &mut EventQueue<TestWaylandApp>,
        app: &mut TestWaylandApp,
    ) {
        let done = Arc::new(AtomicBool::new(false));
        let _cb = conn.display().sync(&queue.handle(), done.clone());
        for _ in 0..32 {
            conn.flush().unwrap();
            display.dispatch_clients(server).unwrap();
            display.flush_clients().unwrap();
            if let Some(guard) = queue.prepare_read() {
                _ = guard.read();
            }
            queue.dispatch_pending(app).unwrap();
            if done.load(Ordering::Acquire) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("roundtrip_app timed out");
    }

    fn server_surface_id(state: &WprsServerState, surface: &ClientWlSurface) -> WlSurfaceId {
        let proto_id = surface.id().protocol_id();
        state
            .object_map
            .iter()
            .find_map(|(&wid, oid)| (oid.protocol_id() == proto_id).then_some(wid))
            .expect("committed surface present in object_map")
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_keyboard_reconnect_enter_modifiers_repeat_and_dedup_wire() {
        let sock_path = std::env::temp_dir().join(format!(
            "wprs_kbd_wire_test_{}_{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut serializer: Serializer<Request, Event> =
            Serializer::new_server(&sock_path).unwrap();
        let reader = serializer.reader().unwrap();
        let mut display: Display<WprsServerState> = Display::new().unwrap();
        let event_loop: EventLoop<WprsServerState> = EventLoop::try_new().unwrap();
        let mut state = WprsServerState::new(
            display.handle(),
            event_loop.handle(),
            serializer,
            false,
            Duration::from_millis(16),
            false,
        );
        state
            .seat
            .add_keyboard(Default::default(), 200, 200)
            .unwrap();
        state.seat.add_pointer();

        // Connect a real Wayland app client to `wprsd`.
        let (server_sock, client_sock) = UnixStream::pair().unwrap();
        display
            .handle()
            .insert_client(
                server_sock,
                Arc::new(ClientState::new(state.serializer.writer())),
            )
            .unwrap();
        let conn = Connection::from_socket(client_sock).unwrap();
        let mut queue: EventQueue<TestWaylandApp> = conn.new_event_queue();
        let qh = queue.handle();
        let mut app = TestWaylandApp::default();
        let _reg = conn.display().get_registry(&qh, ());
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        assert!(app.keyboard.is_some());
        assert!(app.pointer.is_some());

        let compositor = app.compositor.as_ref().unwrap();
        let surf_a = compositor.create_surface(&qh, ());
        surf_a.commit();
        let surf_b = compositor.create_surface(&qh, ());
        surf_b.commit();
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);

        let wid_a = server_surface_id(&state, &surf_a);
        let wid_b = server_surface_id(&state, &surf_b);
        let pid_a = surf_a.id().protocol_id();
        let pid_b = surf_b.id().protocol_id();

        // Activate `wprsc` session 1.
        let wprsc1 = UnixStream::connect(&sock_path).unwrap();
        let (conn1, msg1) = reader.recv().unwrap();
        assert_matches!(msg1, RecvType::Connected);
        state.handle_event(conn1, msg1);

        // Compile a 2-layout keymap where evdev keycode 58 (CapsLock) is remapped
        // to BackSpace so any synthetic keycode-58 injection would immediately show up.
        let xkb_ctx = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_names(
            &xkb_ctx,
            "",
            "",
            "us,de",
            "",
            Some("caps:backspace".to_string()),
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .expect("compile us,de caps:backspace keymap");
        let keymap_str = keymap.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1);
        let caps_bit = 1u32 << keymap.mod_get_index(xkb::MOD_NAME_CAPS);
        let num_bit = 1u32 << keymap.mod_get_index(xkb::MOD_NAME_NUM);
        let ctrl_bit = 1u32 << keymap.mod_get_index(xkb::MOD_NAME_CTRL);
        let alt_bit = 1u32 << keymap.mod_get_index(xkb::MOD_NAME_ALT);

        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Keymap(
                keymap_str.clone(),
            ))),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        app.kbd_events.clear();

        // 1. Unknown-surface Enter must NOT leak keys into `pressed_keys` or Smithay.
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Enter {
                serial: 1,
                surface_id: WlSurfaceId(999_999),
                keycodes: vec![59, 14],
                keysyms: Vec::new(),
            })),
        );
        assert!(state.pressed_keys.is_empty());
        assert!(state.seat.get_keyboard().unwrap().pressed_keys().is_empty());

        // 2. Enter surf_a, verify RepeatInfo::Repeat and RepeatInfo::Disable.
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Enter {
                serial: 2,
                surface_id: wid_a,
                keycodes: Vec::new(),
                keysyms: Vec::new(),
            })),
        );
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::RepeatInfo(
                RepeatInfo::Repeat {
                    rate: NonZeroU32::new(25).unwrap(),
                    delay: 400,
                },
            ))),
        );
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::RepeatInfo(
                RepeatInfo::Disable,
            ))),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        assert!(app.kbd_events.contains(&KbdWireEvent::RepeatInfo {
            rate: 25,
            delay: 400,
        }));
        assert!(
            app.kbd_events
                .contains(&KbdWireEvent::RepeatInfo { rate: 0, delay: 0 })
        );
        app.kbd_events.clear();

        // 3. Modifiers sync updates depressed/locked + layout 1 with ZERO fake
        //    KEY_CAPSLOCK (58) / KEY_NUMLOCK (69) keypresses.
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Modifiers {
                serial: 3,
                raw_modifiers: RawModifiers {
                    depressed: ctrl_bit | alt_bit,
                    latched: 0,
                    locked: caps_bit | num_bit,
                },
                layout_index: 1,
            })),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        assert!(
            !app.kbd_events
                .iter()
                .any(|e| matches!(e, KbdWireEvent::Key { .. })),
            "Modifiers sync must never emit synthetic key events: {:?}",
            app.kbd_events
        );
        assert_eq!(
            app.kbd_events.last(),
            Some(&KbdWireEvent::Modifiers {
                depressed: ctrl_bit | alt_bit,
                latched: 0,
                locked: caps_bit | num_bit,
                group: 1,
            })
        );
        app.kbd_events.clear();

        // Clearing back to layout 0 and zero modifiers notifies focused client cleanly.
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Modifiers {
                serial: 4,
                raw_modifiers: RawModifiers::default(),
                layout_index: 0,
            })),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        assert!(
            !app.kbd_events
                .iter()
                .any(|e| matches!(e, KbdWireEvent::Key { .. }))
        );
        assert_eq!(
            app.kbd_events.last(),
            Some(&KbdWireEvent::Modifiers {
                depressed: 0,
                latched: 0,
                locked: 0,
                group: 0,
            })
        );
        app.kbd_events.clear();

        // 4. Duplicate Pressed/Released deduplication + Repeated handling for Backspace (14).
        for (serial, key_state) in [
            (5, KeyState::Pressed),
            (6, KeyState::Pressed),
            (7, KeyState::Repeated),
            (8, KeyState::Released),
            (9, KeyState::Released),
        ] {
            state.handle_event(
                conn1,
                RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                    serial,
                    raw_code: 14,
                    state: key_state,
                }))),
            );
        }
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        let key_events: Vec<_> = app
            .kbd_events
            .iter()
            .filter(|e| matches!(e, KbdWireEvent::Key { .. }))
            .cloned()
            .collect();
        assert_eq!(
            key_events,
            vec![
                KbdWireEvent::Key {
                    key: 14,
                    pressed: true,
                },
                KbdWireEvent::Key {
                    key: 14,
                    pressed: false,
                },
                KbdWireEvent::Key {
                    key: 14,
                    pressed: true,
                },
                KbdWireEvent::Key {
                    key: 14,
                    pressed: false,
                },
            ]
        );
        assert!(state.pressed_keys.is_empty());
        app.kbd_events.clear();

        // 5. Enter surf_b while surf_a is still focused (missed Leave) with key 59 held:
        //    surf_a must receive Leave (NOT a spurious Key(14, Pressed)), and surf_b
        //    must receive Enter with ONLY `[14]`, not stale `59`.
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                serial: 10,
                raw_code: 59, // KEY_F1
                state: KeyState::Pressed,
            }))),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        app.kbd_events.clear();

        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Enter {
                serial: 11,
                surface_id: wid_b,
                keycodes: vec![14],
                keysyms: Vec::new(),
            })),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        assert!(
            !app.kbd_events
                .iter()
                .any(|e| matches!(e, KbdWireEvent::Key { .. })),
            "switching focus via Enter must not emit Key events to the old surface: {:?}",
            app.kbd_events
        );
        assert!(
            app.kbd_events
                .contains(&KbdWireEvent::Leave { surface_id: pid_a })
        );
        assert!(app.kbd_events.contains(&KbdWireEvent::Enter {
            surface_id: pid_b,
            keys: vec![14],
        }));
        app.kbd_events.clear();

        // 6. Disconnect while surf_b is focused with F1 (59), Backspace (14),
        //    LeftShift (42), CapsLock + layout 1, and a pointer button pressed;
        //    reconnecting `wprsc2` must reset all input state and allow clean
        //    re-entry onto the SAME surf_b with working F1 & Backspace.
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                serial: 12,
                raw_code: 59,
                state: KeyState::Pressed,
            }))),
        );
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                serial: 13,
                raw_code: 42,
                state: KeyState::Pressed,
            }))),
        );
        state.handle_event(
            conn1,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Modifiers {
                serial: 14,
                raw_modifiers: RawModifiers {
                    depressed: 0,
                    latched: 0,
                    locked: caps_bit,
                },
                layout_index: 1,
            })),
        );
        state.handle_event(
            conn1,
            RecvType::Object(Event::PointerFrame(vec![
                PointerEvent {
                    surface_id: wid_b,
                    position: (10.0, 20.0).into(),
                    kind: PointerEventKind::Enter { serial: 15 },
                },
                PointerEvent {
                    surface_id: wid_b,
                    position: (10.0, 20.0).into(),
                    kind: PointerEventKind::Press {
                        button: 0x110,
                        serial: 16,
                    },
                },
            ])),
        );
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);
        app.kbd_events.clear();
        app.pointer_leaves.clear();
        app.pointer_button_releases.clear();

        // Sudden disconnect (no Leave / Release) + reconnect.
        drop(wprsc1);
        let wprsc2 = UnixStream::connect(&sock_path).unwrap();
        let (conn2, msg2) = reader.recv().unwrap();
        assert_matches!(msg2, RecvType::Connected);
        state.handle_event(conn2, msg2);
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);

        assert!(
            app.kbd_events
                .contains(&KbdWireEvent::Leave { surface_id: pid_b }),
            "reconnect must leave previously focused keyboard surface: {:?}",
            app.kbd_events
        );
        assert_eq!(app.pointer_button_releases, vec![0x110]);
        assert_eq!(app.pointer_leaves, vec![pid_b]);
        assert!(state.pressed_keys.is_empty());
        assert!(state.pressed_buttons.is_empty());
        let kbd = state.seat.get_keyboard().unwrap();
        assert!(kbd.pressed_keys().is_empty());
        assert!(kbd.current_focus().is_none());
        assert_eq!(kbd.modifier_state().serialized.depressed, 0);
        assert_eq!(kbd.modifier_state().serialized.locked, 0);
        assert_eq!(kbd.modifier_state().serialized.layout_effective, 0);
        app.kbd_events.clear();

        // New session sends Keymap + Enter on the SAME surf_b -> must deliver fresh
        // Enter with empty keys and exact F1 (59) & Backspace (14) keycodes.
        state.handle_event(
            conn2,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Keymap(keymap_str))),
        );
        state.handle_event(
            conn2,
            RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Enter {
                serial: 20,
                surface_id: wid_b,
                keycodes: Vec::new(),
                keysyms: Vec::new(),
            })),
        );
        for (serial, code, key_state) in [
            (21, 59, KeyState::Pressed),
            (22, 59, KeyState::Released),
            (23, 14, KeyState::Pressed),
            (24, 14, KeyState::Released),
        ] {
            state.handle_event(
                conn2,
                RecvType::Object(Event::KeyboardEvent(KeyboardEvent::Key(KeyInner {
                    serial,
                    raw_code: code,
                    state: key_state,
                }))),
            );
        }
        roundtrip_app(&mut display, &mut state, &conn, &mut queue, &mut app);

        assert!(app.kbd_events.contains(&KbdWireEvent::Enter {
            surface_id: pid_b,
            keys: Vec::new(),
        }));
        let post_reconnect_keys: Vec<_> = app
            .kbd_events
            .iter()
            .filter(|e| matches!(e, KbdWireEvent::Key { .. }))
            .cloned()
            .collect();
        assert_eq!(
            post_reconnect_keys,
            vec![
                KbdWireEvent::Key {
                    key: 59,
                    pressed: true,
                },
                KbdWireEvent::Key {
                    key: 59,
                    pressed: false,
                },
                KbdWireEvent::Key {
                    key: 14,
                    pressed: true,
                },
                KbdWireEvent::Key {
                    key: 14,
                    pressed: false,
                },
            ]
        );

        drop(wprsc2);
        _ = std::fs::remove_file(&sock_path);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_set_compression_mode_and_connect_reset() {
        let sock_path = std::env::temp_dir().join(format!(
            "wprs_comp_mode_test_{}_{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut serializer: Serializer<Request, Event> =
            Serializer::new_server(&sock_path).unwrap();
        let reader = serializer.reader().unwrap();
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
        assert_eq!(state.compression_mode, CompressionMode::Lossless);

        // Client 1 connects -> activates session -> sets Hybrid mode.
        let client1 = UnixStream::connect(&sock_path).unwrap();
        let (conn1, msg1) = reader.recv().unwrap();
        assert_matches!(msg1, RecvType::Connected);
        state.handle_event(conn1, msg1);
        assert_eq!(state.compression_mode, CompressionMode::Lossless);

        let hybrid = CompressionMode::Hybrid {
            quality: 85,
            subsamp: JpegSubsamp::Sub420,
        };
        state.handle_event(conn1, RecvType::Object(Event::SetCompressionMode(hybrid)));
        assert_eq!(state.compression_mode, hybrid);

        // Invalid wire quality (`0` and `101`) must be rejected without mutating state.
        for bad_q in [0u8, 101] {
            let bad = CompressionMode::Hybrid {
                quality: bad_q,
                subsamp: JpegSubsamp::Sub444,
            };
            assert!(state.handle_set_compression_mode(bad).is_err());
            state.handle_event(conn1, RecvType::Object(Event::SetCompressionMode(bad)));
            assert_eq!(state.compression_mode, hybrid);
        }

        // Client 1 disconnects and Client 2 connects -> `handle_connect` resets to Lossless.
        drop(client1);
        let client2 = UnixStream::connect(&sock_path).unwrap();
        let (conn2, msg2) = reader.recv().unwrap();
        assert_matches!(msg2, RecvType::Connected);
        state.handle_event(conn2, msg2);
        assert_eq!(state.compression_mode, CompressionMode::Lossless);

        // Stale event from Client 1 is ignored.
        state.handle_event(conn1, RecvType::Object(Event::SetCompressionMode(hybrid)));
        assert_eq!(state.compression_mode, CompressionMode::Lossless);

        drop(client2);
        _ = std::fs::remove_file(&sock_path);
    }
}
