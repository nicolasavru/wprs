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

/// Handlers for events from smithay client toolkit.
use smithay::reexports::wayland_protocols::wp::primary_selection::zv1::client::zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1;
use smithay::reexports::wayland_protocols::wp::primary_selection::zv1::client::zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewport;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewporter::WpViewporter;
use smithay_client_toolkit::compositor::CompositorHandler;
use smithay_client_toolkit::compositor::SurfaceData;
use smithay_client_toolkit::data_device_manager::data_device::DataDeviceHandler;
use smithay_client_toolkit::data_device_manager::data_offer::DataOfferHandler;
use smithay_client_toolkit::data_device_manager::data_offer::DragOffer;
use smithay_client_toolkit::data_device_manager::data_source::DataSourceHandler;
use smithay_client_toolkit::data_device_manager::WritePipe;
use smithay_client_toolkit::output::OutputData;
use smithay_client_toolkit::output::OutputHandler;
use smithay_client_toolkit::output::OutputState;
use smithay_client_toolkit::primary_selection::device::PrimarySelectionDeviceHandler;
use smithay_client_toolkit::primary_selection::selection::PrimarySelectionSourceHandler;
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
use smithay_client_toolkit::seat::pointer::PointerEvent;
use smithay_client_toolkit::seat::pointer::PointerEventKind;
use smithay_client_toolkit::seat::pointer::PointerHandler;
use smithay_client_toolkit::seat::pointer::ThemeSpec;
use smithay_client_toolkit::seat::Capability;
use smithay_client_toolkit::seat::SeatHandler;
use smithay_client_toolkit::seat::SeatState;
use smithay_client_toolkit::shell::xdg::popup;
use smithay_client_toolkit::shell::xdg::window::Window;
use smithay_client_toolkit::shell::xdg::window::WindowConfigure;
use smithay_client_toolkit::shell::xdg::window::WindowHandler;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shm::Shm;
use smithay_client_toolkit::shm::ShmHandler;
use tracing::Span;

use crate::args;
use crate::client::RemoteClient;
use crate::client::Role;
use crate::client::SeatObject;
use crate::client::WprsClientState;
use crate::client::subsurface;
use crate::prelude::*;
use crate::serialization::Event;
use crate::serialization::SendType;
use crate::serialization::wayland;
use crate::serialization::wayland::DataDestinationEvent;
use crate::serialization::wayland::DataEvent;
use crate::serialization::wayland::DataSource;
use crate::serialization::wayland::DataSourceEvent;
use crate::serialization::wayland::DataTransferTarget;
use crate::serialization::wayland::DragEnter;
use crate::serialization::wayland::KeyInner;
use crate::serialization::wayland::KeyState;
use crate::serialization::wayland::KeyboardEvent;
use crate::serialization::wayland::Output;
use crate::serialization::wayland::OutputEvent;
use crate::serialization::wayland::SourceMetadata;
use crate::serialization::wayland::SurfaceEvent;
use crate::serialization::wayland::SurfaceEventPayload::OutputsChanged;
use crate::serialization::wayland::WlSurfaceId;
use crate::serialization::xdg_shell::PopupConfigure;
use crate::serialization::xdg_shell::PopupEvent;
use crate::serialization::xdg_shell::ToplevelClose;
use crate::serialization::xdg_shell::ToplevelConfigure;
use crate::serialization::xdg_shell::ToplevelEvent;

impl WprsClientState {
    fn send_surface_outputs(&self, surface: &WlSurface) {
        let Some(&(_, surface_id)) = self.object_bimap.get_by_right(&surface.id()) else {
            return;
        };
        let outputs = surface.data::<SurfaceData<()>>().map(SurfaceData::outputs);

        if let Some(outputs) = outputs {
            let outputs = outputs
                .filter_map(|output| {
                    output.data::<OutputData>().map(|data| Output {
                        id: data.with_output_info(|info| info.id),
                    })
                })
                .collect();

            self.serializer
                .writer()
                .send(SendType::Object(Event::Surface(SurfaceEvent {
                    surface_id,
                    payload: OutputsChanged(outputs),
                })));
        }
    }
}

fn commit_children_and_draw(
    client: &mut RemoteClient,
    surface_id: WlSurfaceId,
    qh: &QueueHandle<WprsClientState>,
) {
    subsurface::commit_sync_children(surface_id, &mut client.surfaces).log_and_ignore(loc!());
    client
        .surface(&surface_id)
        .and_then(|surface| surface.draw_buffer_send_frame(qh))
        .log_and_ignore(loc!());
}

impl CompositorHandler for WprsClientState {
    #[instrument(skip(self, _conn, _qh, _surface, _new_factor), level = "debug")]
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &WlSurface,
        _new_factor: i32,
    ) {
        // Surface output membership is forwarded by `surface_enter`/`surface_leave`.
    }

    #[instrument(skip(self, _conn, _qh, _surface, _new_transform), level = "debug")]
    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &WlSurface,
        _new_transform: Transform,
    ) {
        // Surface output membership is forwarded by `surface_enter`/`surface_leave`.
    }

    #[instrument(skip(self, _conn, qh, _time), level = "debug")]
    fn frame(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _time: u32,
    ) {
        let Some(&(client_id, surface_id)) = self.object_bimap.get_by_right(&surface.id()) else {
            return;
        };
        let Some(client) = self.remote_display.clients.get_mut(&client_id) else {
            return;
        };

        subsurface::commit_sync_children(surface_id, &mut client.surfaces).log_and_ignore(loc!());

        let Ok(surface) = client.surface(&surface_id) else {
            return;
        };

        surface.frame_callback_completed = true;
        match &surface.role {
            Some(Role::SubSurface(subsurface)) if subsurface.sync => {},
            _ => {
                surface.draw_buffer_send_frame(qh).log_and_ignore(loc!());
            },
        }
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _output: &WlOutput,
    ) {
        self.send_surface_outputs(surface);
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _output: &WlOutput,
    ) {
        self.send_surface_outputs(surface);
    }
}

impl OutputHandler for WprsClientState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn new_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: WlOutput) {
        let output_info = self.output_state().info(&output).unwrap();
        self.serializer
            .writer()
            .send(SendType::Object(Event::Output(OutputEvent::New(
                output_info.into(),
            ))));
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn update_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: WlOutput) {
        let output_info = self.output_state().info(&output).unwrap();
        self.serializer
            .writer()
            .send(SendType::Object(Event::Output(OutputEvent::Update(
                output_info.into(),
            ))));
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn output_destroyed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: WlOutput) {
        let output_info = self.output_state().info(&output).unwrap();
        self.serializer
            .writer()
            .send(SendType::Object(Event::Output(OutputEvent::Destroy(
                output_info.into(),
            ))));
    }
}

impl WindowHandler for WprsClientState {
    #[instrument(skip_all, level = "debug")]
    fn request_close(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, window: &Window) {
        let Some(&(_, surface_id)) = self.object_bimap.get_by_right(&window.wl_surface().id())
        else {
            return;
        };

        self.serializer
            .writer()
            .send(SendType::Object(Event::Toplevel(ToplevelEvent::Close(
                ToplevelClose { surface_id },
            ))));
    }

    #[instrument(skip_all, level = "debug")]
    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        window: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        let Some(&(client_id, surface_id)) =
            self.object_bimap.get_by_right(&window.wl_surface().id())
        else {
            return;
        };
        let Some(client) = self.remote_display.clients.get_mut(&client_id) else {
            return;
        };
        let Ok(surface) = client.surface(&surface_id) else {
            return;
        };
        let Some(toplevel) = surface.role.as_mut().and_then(Role::as_xdg_toplevel_mut) else {
            return;
        };
        if !std::mem::replace(&mut toplevel.configured, true) {
            commit_children_and_draw(client, surface_id, qh);
        }

        self.serializer
            .writer()
            .send(SendType::Object(Event::Toplevel(ToplevelEvent::Configure(
                ToplevelConfigure::from_smithay(&surface_id, configure),
            ))));
    }
}

impl popup::PopupHandler for WprsClientState {
    #[instrument(skip_all, level = "debug")]
    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        popup: &popup::Popup,
        configure: popup::PopupConfigure,
    ) {
        let Some(&(client_id, surface_id)) =
            self.object_bimap.get_by_right(&popup.wl_surface().id())
        else {
            return;
        };
        let Some(client) = self.remote_display.clients.get_mut(&client_id) else {
            return;
        };
        let Ok(surface) = client.surface(&surface_id) else {
            return;
        };
        let Some(remote_popup) = surface.role.as_mut().and_then(Role::as_xdg_popup_mut) else {
            return;
        };
        if !std::mem::replace(&mut remote_popup.configured, true) {
            commit_children_and_draw(client, surface_id, qh);
        }

        self.serializer
            .writer()
            .send(SendType::Object(Event::Popup(PopupEvent::Configure(
                PopupConfigure::from_smithay(&surface_id, configure),
            ))));
    }

    fn done(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, popup: &popup::Popup) {
        let Some(&(_, surface_id)) = self.object_bimap.get_by_right(&popup.wl_surface().id())
        else {
            return;
        };
        self.serializer
            .writer()
            .send(SendType::Object(Event::Popup(PopupEvent::Done(surface_id))));
    }
}

impl SeatHandler for WprsClientState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlSeat) {}

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: WlSeat,
        capability: Capability,
    ) {
        let seat_obj = if let Some(seat_obj) = self.seat_objects.iter_mut().find(|s| s.seat == seat)
        {
            seat_obj
        } else {
            // create the data device here for this seat
            let data_device_manager = &self.data_device_manager_state;
            let data_device = data_device_manager.get_data_device(qh, &seat);

            let primary_selection_device = self.primary_selection_manager_state.as_ref().map(
                |primary_selection_manager_state| {
                    primary_selection_manager_state.get_selection_device(qh, &seat)
                },
            );

            self.seat_objects.push(SeatObject {
                seat: seat.clone(),
                keyboard: None,
                pointer: None,
                data_device,
                primary_selection_device,
            });
            self.seat_objects.last_mut().unwrap()
        };

        if capability == Capability::Keyboard && seat_obj.keyboard.is_none() {
            debug!("set keyboard capability");
            let keyboard = self
                .seat_state
                .get_keyboard(qh, &seat, None)
                .expect("Failed to create keyboard");
            seat_obj.keyboard.replace(keyboard);
        }

        if capability == Capability::Pointer && seat_obj.pointer.is_none() {
            debug!("set pointer capability");
            let themed_pointer = self
                .seat_state
                .get_pointer_with_theme::<_, ()>(
                    qh,
                    &seat,
                    self.shm_state.wl_shm(),
                    self.compositor_state.create_surface(qh),
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
        if let Some(seat_obj) = self.seat_objects.iter_mut().find(|s| s.seat == seat) {
            match capability {
                Capability::Keyboard if let Some(k) = seat_obj.keyboard.take() => {
                    k.release();
                },
                Capability::Pointer => {
                    seat_obj.pointer.take();
                },
                _ => {},
            }
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlSeat) {}
}

impl KeyboardHandler for WprsClientState {
    #[instrument(skip(self, _conn, _qh, _keyboard), level = "debug")]
    fn enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        surface: &WlSurface,
        serial: u32,
        raw: &[u32],
        keysyms: &[Keysym],
    ) {
        let Some(&(_, surface_id)) = self.object_bimap.get_by_right(&surface.id()) else {
            // TODO: unwrap is wrong, we can enter before surface exists.
            // Currently we're just returning in that case, but should we create
            // a surface instead?
            return;
        };

        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(
                KeyboardEvent::Enter {
                    serial,
                    surface_id,
                    keycodes: raw.into(),
                    keysyms: keysyms.iter().map(|k| k.raw()).collect(),
                },
            )));
    }

    #[instrument(skip(self, _conn, _qh, _keyboard), level = "debug")]
    fn leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        surface: &WlSurface,
        serial: u32,
    ) {
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(
                KeyboardEvent::Leave { serial },
            )));
    }

    // INTENTIONALLY NOT LOGGING KEY EVENTS
    #[instrument(
        skip(self, _conn, _qh, _keyboard, event),
        fields(event = "<redacted>"),
        level = "debug"
    )]
    fn press_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        self.last_implicit_grab_serial = Some(serial);
        if args::get_log_priv_data() {
            Span::current().record("event", field::debug(&event));
        }
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(KeyboardEvent::Key(
                KeyInner {
                    serial,
                    raw_code: event.raw_code,
                    state: KeyState::Pressed,
                },
            ))));
    }

    // INTENTIONALLY NOT LOGGING KEY EVENTS
    #[instrument(
        skip(self, _conn, _qh, _keyboard, serial, event),
        fields(event = "<redacted>"),
        level = "debug"
    )]
    fn repeat_key(
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
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(KeyboardEvent::Key(
                KeyInner {
                    serial,
                    raw_code: event.raw_code,
                    state: KeyState::Repeated,
                },
            ))));
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
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(KeyboardEvent::Key(
                KeyInner {
                    serial,
                    raw_code: event.raw_code,
                    state: KeyState::Released,
                },
            ))));
    }

    #[instrument(skip(self, _conn, _qh, _keyboard), level = "debug")]
    fn update_repeat_info(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        info: RepeatInfo,
    ) {
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(
                KeyboardEvent::RepeatInfo(info.into()),
            )));
    }

    #[instrument(skip(self, _conn, _qh, _keyboard, keymap), level = "debug")]
    fn update_keymap(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        keymap: Keymap<'_>,
    ) {
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(
                KeyboardEvent::Keymap(keymap.as_string()),
            )));
    }

    #[instrument(skip(self, _conn, _qh, _keyboard, _modifiers), level = "debug")]
    fn update_modifiers(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        serial: u32,
        _modifiers: Modifiers,
        raw_modifiers: RawModifiers,
        variant: u32,
    ) {
        self.serializer
            .writer()
            .send(SendType::Object(Event::KeyboardEvent(
                KeyboardEvent::Modifiers {
                    serial,
                    raw_modifiers: raw_modifiers.into(),
                    layout_index: variant,
                },
            )));
    }
}

impl PointerHandler for WprsClientState {
    #[instrument(skip(self, _conn, _qh, _pointer), level = "debug")]
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _pointer: &WlPointer,
        events: &[PointerEvent],
    ) {
        let has_enter = events
            .iter()
            .any(|event| matches!(event.kind, PointerEventKind::Enter { .. }));
        let mut serialized_events = Vec::with_capacity(events.len());
        for event in events {
            let Some(&(_, surface_id)) = self.object_bimap.get_by_right(&event.surface.id()) else {
                continue;
            };

            match event.kind {
                PointerEventKind::Enter { serial } => {
                    self.last_enter_serial = serial;
                    self.apply_current_cursor_image().log_and_ignore(loc!());
                },
                PointerEventKind::Leave { .. } => {
                    if self.dnd_source.is_none() && !has_enter {
                        self.pressed_buttons.clear();
                    }
                },
                PointerEventKind::Press { button, serial, .. } => {
                    self.last_mouse_down_serial = Some(serial);
                    self.pressed_buttons.insert(button);
                },
                PointerEventKind::Release { button, .. } => {
                    self.pressed_buttons.remove(&button);
                    // A host compositor accepting `start_drag` grabs the
                    // pointer and sends `wl_pointer.leave` instead of button
                    // release events. Seeing the final release while
                    // `dnd_source` is undropped means the host denied
                    // `start_drag` (e.g., button released in flight).
                    if self.pressed_buttons.is_empty()
                        && !self.dnd_source_dropped
                        && let Some((source_id, _)) = self.dnd_source.take()
                    {
                        self.serializer.writer().send(SendType::Object(Event::Data(
                            DataEvent::SourceEvent(DataSourceEvent::DnDCancelled { source_id }),
                        )));
                    }
                },
                _ => {},
            }
            serialized_events.push(wayland::PointerEvent::from_smithay(&surface_id, event));
        }

        if !serialized_events.is_empty() {
            self.serializer
                .writer()
                .send(SendType::Object(Event::PointerFrame(serialized_events)));
        }
    }
}

impl ShmHandler for WprsClientState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm_state
    }
}

impl DataDeviceHandler for WprsClientState {
    #[instrument(skip_all, level = "debug")]
    fn enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        data_device: &WlDataDevice,
        _x: f64,
        _y: f64,
        _wl_surface: &WlSurface,
    ) {
        if let Some((old_offer_id, _)) = self.active_dnd_offer.take() {
            self.serializer.writer().send(SendType::Object(Event::Data(
                DataEvent::DestinationEvent(DataDestinationEvent::DnDLeave {
                    offer_id: old_offer_id,
                }),
            )));
        }
        // SCTK destroys any previous `drag_offer` when handling `Event::Enter`.
        self.dropped_dnd_offers.clear();

        let data_device = &self
            .seat_objects
            .iter()
            .find(|seat| seat.data_device.inner() == data_device)
            .unwrap()
            .data_device
            .data();
        let Some(drag_offer) = data_device.drag_offer() else {
            return;
        };
        debug!(
            "data offer entered x: {:.2} y: {:.2}",
            drag_offer.x, drag_offer.y
        );
        let mime_types = drag_offer.with_mime_types(<[String]>::to_vec);
        let Some(&(_, surface_id)) = self.object_bimap.get_by_right(&drag_offer.surface.id())
        else {
            return;
        };
        let offer_id = self.next_dnd_offer_id;
        self.next_dnd_offer_id += 1;
        self.active_dnd_offer = Some((offer_id, drag_offer.clone()));
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::DnDEnter(DragEnter {
                    offer_id,
                    serial: drag_offer.serial,
                    surface: surface_id,
                    loc: (drag_offer.x, drag_offer.y).into(),
                    source_actions: drag_offer.source_actions.into(),
                    selected_action: drag_offer.selected_action.into(),
                    mime_types,
                }),
            ))));
    }

    #[instrument(skip_all, level = "debug")]
    fn leave(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _data_device: &WlDataDevice) {
        debug!("data offer left");
        if let Some((offer_id, _)) = self.active_dnd_offer.take() {
            self.serializer.writer().send(SendType::Object(Event::Data(
                DataEvent::DestinationEvent(DataDestinationEvent::DnDLeave { offer_id }),
            )));
        }
    }

    #[instrument(skip_all, level = "debug")]
    fn motion(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _data_device: &WlDataDevice,
        x: f64,
        y: f64,
    ) {
        let Some((offer_id, _)) = &self.active_dnd_offer else {
            return;
        };
        debug!("data offer motion x: {:.2} y: {:.2}", x, y);
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::DnDMotion {
                    offer_id: *offer_id,
                    loc: (x, y).into(),
                },
            ))));
    }

    #[instrument(skip_all, level = "debug")]
    fn selection(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        data_device: &WlDataDevice,
    ) {
        let data_device = &self
            .seat_objects
            .iter()
            .find(|seat| seat.data_device.inner() == data_device)
            .unwrap()
            .data_device
            .data();
        let Some(offer) = data_device.selection_offer() else {
            return;
        };
        let mime_types = offer.with_mime_types(<[String]>::to_vec);
        if mime_types.contains(&"_wprs_marker".to_string()) {
            return;
        }
        self.selection_offer = Some(offer);
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::SelectionSet(
                    DataSource::Selection,
                    SourceMetadata::from_mime_types(mime_types),
                ),
            ))));
    }

    #[instrument(skip_all, level = "debug")]
    fn drop_performed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        data_device: &WlDataDevice,
    ) {
        let Some((offer_id, mut offer)) = self.active_dnd_offer.take() else {
            return;
        };
        if let Some(updated) = self
            .seat_objects
            .iter()
            .find(|seat| seat.data_device.inner() == data_device)
            .and_then(|seat| seat.data_device.data().drag_offer())
        {
            offer = updated;
        } else {
            offer.dropped = true;
        }
        self.dropped_dnd_offers.insert(offer_id, offer);
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::DnDDrop { offer_id },
            ))));
    }
}

impl DataOfferHandler for WprsClientState {
    #[instrument(skip(self, _conn, _qh, offer), level = "debug")]
    fn source_actions(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        offer: &mut DragOffer,
        actions: DndAction,
    ) {
        if let Some(offer_id) = self.dnd_offer_id_for(offer) {
            self.serializer.writer().send(SendType::Object(Event::Data(
                DataEvent::DestinationEvent(DataDestinationEvent::DnDSourceActions {
                    offer_id,
                    actions: actions.into(),
                }),
            )));
        }
    }

    #[instrument(skip(self, _conn, _qh, offer), level = "debug")]
    fn selected_action(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        offer: &mut DragOffer,
        actions: DndAction,
    ) {
        if let Some(offer_id) = self.dnd_offer_id_for(offer) {
            self.serializer.writer().send(SendType::Object(Event::Data(
                DataEvent::DestinationEvent(DataDestinationEvent::DnDActionSelected {
                    offer_id,
                    action: actions.into(),
                }),
            )));
        }
    }
}

impl DataSourceHandler for WprsClientState {
    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn accept_mime(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &WlDataSource,
        mime: Option<String>,
    ) {
        if let Some((source_id, dnd_source)) = &self.dnd_source
            && source == dnd_source.inner()
        {
            self.serializer
                .writer()
                .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::DnDMimeTypeAcceptedByDestination {
                        source_id: *source_id,
                        mime_type: mime,
                    },
                ))));
        }
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
        let target = match (&self.selection_source, &self.dnd_source) {
            (Some(selection_source), _) if source == selection_source.inner() => {
                DataTransferTarget::Selection
            },
            (_, Some((source_id, dnd_source))) if source == dnd_source.inner() => {
                DataTransferTarget::DnD(*source_id)
            },
            _ => {
                warn!("request for unknown source");
                return;
            },
        };
        let transfer_id = self.allocate_transfer_id();
        self.pending_send_pipes.insert(transfer_id, write_pipe);
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                DataSourceEvent::MimeTypeSendRequestedByDestination {
                    target,
                    mime_type: mime,
                    transfer_id,
                },
            ))));
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn cancelled(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, source: &WlDataSource) {
        if self
            .selection_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            // Dropping `CopyPasteSource` destroys the underlying `WlDataSource`.
            self.selection_source = None;
        } else if self
            .dnd_source
            .as_ref()
            .is_some_and(|(_, s)| s.inner() == source)
        {
            let (source_id, _) = self.dnd_source.take().unwrap();
            self.dnd_source_dropped = false;
            self.pressed_buttons.clear();
            self.serializer
                .writer()
                .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::DnDCancelled { source_id },
                ))));
        } else {
            warn!("cancellation for unknown source");
            source.destroy();
        }
    }

    #[instrument(skip_all, level = "debug")]
    fn dnd_dropped(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, source: &WlDataSource) {
        if let Some((source_id, dnd_source)) = &self.dnd_source
            && source == dnd_source.inner()
        {
            let source_id = *source_id;
            self.dnd_source_dropped = true;
            self.pressed_buttons.clear();
            self.serializer
                .writer()
                .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::DnDDropPerformed { source_id },
                ))));
        }
    }

    #[instrument(skip_all, level = "debug")]
    fn dnd_finished(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, source: &WlDataSource) {
        if self
            .dnd_source
            .as_ref()
            .is_some_and(|(_, s)| s.inner() == source)
        {
            let (source_id, _) = self.dnd_source.take().unwrap();
            self.dnd_source_dropped = false;
            self.pressed_buttons.clear();
            self.serializer
                .writer()
                .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::DnDFinished { source_id },
                ))));
        }
    }

    #[instrument(skip(self, _conn, _qh, source), level = "debug")]
    fn action(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &WlDataSource,
        action: DndAction,
    ) {
        if let Some((source_id, dnd_source)) = &self.dnd_source
            && source == dnd_source.inner()
        {
            self.serializer
                .writer()
                .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::DnDActionSelected {
                        source_id: *source_id,
                        action: action.into(),
                    },
                ))));
        }
    }
}

impl PrimarySelectionDeviceHandler for WprsClientState {
    #[instrument(skip_all, level = "debug")]
    fn selection(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        primary_selection_device: &ZwpPrimarySelectionDeviceV1,
    ) {
        if self.primary_selection_manager_state.is_none() {
            return;
        }
        let primary_selection_device = &self
            .seat_objects
            .iter()
            .filter_map(|seat| seat.primary_selection_device.as_ref())
            .find(|device| device.inner() == primary_selection_device)
            .unwrap()
            .data();
        let Some(offer) = primary_selection_device.selection_offer() else {
            return;
        };
        let mime_types = offer.with_mime_types(<[String]>::to_vec);
        if mime_types.contains(&"_wprs_marker".to_string()) {
            return;
        }
        self.primary_selection_offer = Some(offer);
        self.serializer
            .writer()
            .send(SendType::Object(Event::Data(DataEvent::DestinationEvent(
                DataDestinationEvent::SelectionSet(
                    DataSource::Primary,
                    SourceMetadata::from_mime_types(mime_types),
                ),
            ))));
    }
}

impl PrimarySelectionSourceHandler for WprsClientState {
    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn send_request(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &ZwpPrimarySelectionSourceV1,
        mime: String,
        write_pipe: WritePipe,
    ) {
        if self
            .primary_selection_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            let transfer_id = self.allocate_transfer_id();
            self.pending_send_pipes.insert(transfer_id, write_pipe);
            self.serializer
                .writer()
                .send(SendType::Object(Event::Data(DataEvent::SourceEvent(
                    DataSourceEvent::MimeTypeSendRequestedByDestination {
                        target: DataTransferTarget::Primary,
                        mime_type: mime,
                        transfer_id,
                    },
                ))));
        } else {
            warn!("request for unknown source");
        }
    }

    #[instrument(skip(self, _conn, _qh, source), level = "debug")]
    fn cancelled(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &ZwpPrimarySelectionSourceV1,
    ) {
        if self
            .primary_selection_source
            .as_ref()
            .is_some_and(|s| s.inner() == source)
        {
            self.primary_selection_source = None;
        }
    }
}

smithay_client_toolkit::delegate_dispatch2!(WprsClientState);
smithay_client_toolkit::delegate_registry!(WprsClientState);
smithay_client_toolkit::reexports::client::delegate_noop!(WprsClientState: WpViewporter);

impl ProvidesRegistryState for WprsClientState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState,];
}

pub(crate) struct SubCompositorData;

impl Dispatch<WlSubcompositor, SubCompositorData> for WprsClientState {
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

pub(crate) struct SubSurfaceData;

impl Dispatch<WlSubsurface, SubSurfaceData> for WprsClientState {
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

impl Dispatch<WpViewport, ()> for WprsClientState {
    fn event(
        _state: &mut Self,
        _viewport: &WpViewport,
        _event: wp_viewport::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        unreachable!("There are no wp_viewport events")
    }
}
