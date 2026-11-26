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

/// `wl_data_device_manager` v3 implementation for `wprsd` supporting clipboard
/// selection and transparent 1:1 proxied Wayland drag-and-drop through `wprsc`
/// and the host compositor.
use std::collections::HashMap;
use std::mem;
use std::os::fd::AsFd;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use smithay::backend::input::ButtonState;
use smithay::input::Seat;
use smithay::input::SeatHandler;
use smithay::input::pointer::AxisFrame;
use smithay::input::pointer::ButtonEvent;
use smithay::input::pointer::Focus;
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
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::DataInit;
use smithay::reexports::wayland_server::Dispatch;
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::reexports::wayland_server::GlobalDispatch;
use smithay::reexports::wayland_server::New;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::WEnum;
use smithay::reexports::wayland_server::Weak;
use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::reexports::wayland_server::backend::Handle;
use smithay::reexports::wayland_server::backend::ObjectData;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::backend::protocol::Message;
use smithay::reexports::wayland_server::protocol::wl_data_device;
use smithay::reexports::wayland_server::protocol::wl_data_device::WlDataDevice;
use smithay::reexports::wayland_server::protocol::wl_data_device_manager;
use smithay::reexports::wayland_server::protocol::wl_data_device_manager::DndAction;
use smithay::reexports::wayland_server::protocol::wl_data_device_manager::WlDataDeviceManager;
use smithay::reexports::wayland_server::protocol::wl_data_offer;
use smithay::reexports::wayland_server::protocol::wl_data_offer::WlDataOffer;
use smithay::reexports::wayland_server::protocol::wl_data_source;
use smithay::reexports::wayland_server::protocol::wl_data_source::WlDataSource;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Logical;
use smithay::utils::Point;
use smithay::utils::SERIAL_COUNTER;
use smithay::utils::Serial;
use smithay::wayland::compositor;
use smithay::wayland::selection::data_device::DND_ICON_ROLE;
use smithay::wayland::selection::data_device::default_action_chooser;

use crate::prelude::*;
use crate::serialization::Request;
use crate::serialization::SendType;
use crate::serialization::wayland::ClientSurface;
use crate::serialization::wayland::DataDestinationRequest;
use crate::serialization::wayland::DataRequest;
use crate::serialization::wayland::DataSource;
use crate::serialization::wayland::DataSourceRequest;
use crate::serialization::wayland::DataTransferTarget;
use crate::serialization::wayland::DragEnter;
use crate::serialization::wayland::SourceMetadata;
use crate::serialization::wayland::StartDrag;
use crate::server::WprsServerState;

#[derive(Debug)]
struct DataSourceInner {
    mime_types: Vec<String>,
    dnd_actions: DndAction,
}

#[derive(Debug)]
pub struct DataSourceUserData {
    inner: Mutex<DataSourceInner>,
    alive: AtomicBool,
}

impl DataSourceUserData {
    fn new() -> Self {
        Self {
            inner: Mutex::new(DataSourceInner {
                mime_types: Vec::new(),
                dnd_actions: DndAction::None,
            }),
            alive: AtomicBool::new(true),
        }
    }
}

#[derive(Debug)]
pub struct DataDeviceUserData {
    wl_seat: WlSeat,
}

fn source_is_alive(source: &WlDataSource) -> bool {
    source
        .data::<DataSourceUserData>()
        .is_some_and(|data| data.alive.load(Ordering::Acquire))
}

fn source_metadata(source: &WlDataSource) -> SourceMetadata {
    source
        .data::<DataSourceUserData>()
        .map(|data| {
            let guard = data.inner.lock().unwrap();
            let dnd_actions = if source.version() < wl_data_source::REQ_SET_ACTIONS_SINCE {
                DndAction::Copy
            } else {
                guard.dnd_actions
            };
            SourceMetadata {
                mime_types: guard.mime_types.clone(),
                dnd_actions: dnd_actions.into(),
            }
        })
        .unwrap_or_else(|| SourceMetadata::from_mime_types(Vec::new()))
}

fn source_contains_mime(source: &WlDataSource, mime_type: &str) -> bool {
    source.data::<DataSourceUserData>().is_some_and(|data| {
        data.inner
            .lock()
            .unwrap()
            .mime_types
            .iter()
            .any(|m| m == mime_type)
    })
}

fn is_single_dnd_action(action: DndAction) -> bool {
    matches!(
        action,
        DndAction::None | DndAction::Copy | DndAction::Move | DndAction::Ask
    )
}

#[derive(Debug, Clone)]
enum ClipboardSelection {
    Client(WlDataSource),
    Host(Vec<String>),
}

impl ClipboardSelection {
    fn mime_types(&self) -> Vec<String> {
        match self {
            Self::Client(source) => source_metadata(source).mime_types,
            Self::Host(mime_types) => mime_types.clone(),
        }
    }

    fn contains_mime(&self, mime_type: &str) -> bool {
        match self {
            Self::Client(source) => source_contains_mime(source, mime_type),
            Self::Host(mime_types) => mime_types.iter().any(|m| m == mime_type),
        }
    }
}

struct ClipboardOfferData {
    selection: ClipboardSelection,
}

impl ObjectData<WprsServerState> for ClipboardOfferData {
    fn request(
        self: Arc<Self>,
        dh: &Handle,
        state: &mut WprsServerState,
        _client_id: ClientId,
        msg: Message<ObjectId, OwnedFd>,
    ) -> Option<Arc<dyn ObjectData<WprsServerState>>> {
        let dh = DisplayHandle::from(dh.clone());
        let Ok((_offer, wl_data_offer::Request::Receive { mime_type, fd })) =
            WlDataOffer::parse_request(&dh, msg)
        else {
            return None;
        };

        if !self.selection.contains_mime(&mime_type) {
            debug!("Denying wl_data_offer.receive for unknown clipboard mime type {mime_type:?}");
            return None;
        }

        match &self.selection {
            ClipboardSelection::Client(source) => {
                if source_is_alive(source) {
                    source.send(mime_type, fd.as_fd());
                }
            },
            ClipboardSelection::Host(_) => {
                state.request_host_data_transfer(DataTransferTarget::Selection, mime_type, fd);
            },
        }
        None
    }

    fn destroyed(
        self: Arc<Self>,
        _handle: &Handle,
        _data: &mut WprsServerState,
        _client_id: ClientId,
        _object_id: ObjectId,
    ) {
    }
}

#[derive(Debug)]
struct ActiveDndSource {
    id: u64,
    source: WlDataSource,
    /// Set once `DnDDropPerformed` arrives and the pointer drag grab has ended.
    dropped: bool,
}

#[derive(Debug)]
struct DndOfferState {
    offer_id: u64,
    mime_types: Vec<String>,
    source_actions: DndAction,
    host_selected_action: DndAction,
    destination_actions: DndAction,
    preferred_action: DndAction,
    destination_actions_pending: bool,
    active: bool,
    dropped: bool,
    accepted: bool,
    finished: bool,
    live_offer_count: usize,
}

#[derive(Debug)]
struct ActiveIncomingDnd {
    offer_id: u64,
    surface: WlSurface,
    offer_state: Arc<Mutex<DndOfferState>>,
    offers: Vec<WlDataOffer>,
}

#[derive(Debug)]
struct DroppedDndOffer {
    offer_state: Arc<Mutex<DndOfferState>>,
    offers: Vec<WlDataOffer>,
}

struct DndOfferObjectData {
    offer_state: Arc<Mutex<DndOfferState>>,
    destroyed: AtomicBool,
}

impl DndOfferObjectData {
    fn on_destroy(&self, state: &mut WprsServerState) {
        if self.destroyed.swap(true, Ordering::AcqRel) {
            return;
        }
        let cancel_offer_id = {
            let mut offer_state = self.offer_state.lock().unwrap();
            offer_state.live_offer_count = offer_state.live_offer_count.saturating_sub(1);
            if offer_state.live_offer_count == 0
                && offer_state.dropped
                && !offer_state.finished
                && offer_state.active
            {
                offer_state.active = false;
                Some(offer_state.offer_id)
            } else {
                None
            }
        };
        if let Some(offer_id) = cancel_offer_id {
            state.data_device_state.dropped_dnd_offers.remove(&offer_id);
            state
                .serializer
                .writer()
                .send(SendType::Object(Request::Data(
                    DataRequest::DestinationRequest(DataDestinationRequest::DnDCancel { offer_id }),
                )));
        }
    }
}

impl ObjectData<WprsServerState> for DndOfferObjectData {
    fn request(
        self: Arc<Self>,
        dh: &Handle,
        state: &mut WprsServerState,
        _client_id: ClientId,
        msg: Message<ObjectId, OwnedFd>,
    ) -> Option<Arc<dyn ObjectData<WprsServerState>>> {
        let dh = DisplayHandle::from(dh.clone());
        let Ok((offer, request)) = WlDataOffer::parse_request(&dh, msg) else {
            return None;
        };

        match request {
            wl_data_offer::Request::Accept { mime_type, .. } => {
                let req = {
                    let mut offer_state = self.offer_state.lock().unwrap();
                    if !offer_state.active {
                        None
                    } else {
                        let accepted_mime =
                            mime_type.filter(|m| offer_state.mime_types.contains(m));
                        offer_state.accepted = accepted_mime.is_some();
                        Some(DataDestinationRequest::DnDAcceptMimeType {
                            offer_id: offer_state.offer_id,
                            mime_type: accepted_mime,
                        })
                    }
                };
                if let Some(req) = req {
                    state
                        .serializer
                        .writer()
                        .send(SendType::Object(Request::Data(
                            DataRequest::DestinationRequest(req),
                        )));
                }
            },
            wl_data_offer::Request::Receive { mime_type, fd } => {
                let offer_id = {
                    let offer_state = self.offer_state.lock().unwrap();
                    (offer_state.active && offer_state.mime_types.contains(&mime_type))
                        .then_some(offer_state.offer_id)
                };
                if let Some(offer_id) = offer_id {
                    state.request_host_data_transfer(
                        DataTransferTarget::DnD(offer_id),
                        mime_type,
                        fd,
                    );
                }
            },
            wl_data_offer::Request::Destroy => {
                self.on_destroy(state);
            },
            wl_data_offer::Request::Finish => {
                let finish_offer_id = {
                    let mut offer_state = self.offer_state.lock().unwrap();
                    if !offer_state.active {
                        offer.post_error(
                            wl_data_offer::Error::InvalidFinish,
                            "Cannot finish a data offer that is no longer active.",
                        );
                        return None;
                    }
                    if !offer_state.dropped {
                        offer.post_error(
                            wl_data_offer::Error::InvalidFinish,
                            "Cannot finish a data offer that has not been dropped.",
                        );
                        return None;
                    }
                    if !offer_state.accepted {
                        offer.post_error(
                            wl_data_offer::Error::InvalidFinish,
                            "Cannot finish a data offer that has not been accepted.",
                        );
                        return None;
                    }
                    // Allow synchronous post-drop `set_actions(Move/Copy)` +
                    // `finish()` in a single client flush (e.g., Dolphin `Ask`
                    // popup) before the host compositor's `DnDActionSelected`
                    // reply completes its round trip, while still rejecting
                    // post-drop `set_actions(None, None)` + `finish()`.
                    let effective_action = if !offer_state.destination_actions_pending
                        && !offer_state.host_selected_action.is_empty()
                    {
                        offer_state.host_selected_action
                    } else {
                        default_action_chooser(
                            offer_state.source_actions & offer_state.destination_actions,
                            offer_state.preferred_action,
                        )
                    };
                    if effective_action.is_empty() || effective_action == DndAction::Ask {
                        offer.post_error(
                            wl_data_offer::Error::InvalidFinish,
                            "Cannot finish a data offer with no valid action.",
                        );
                        return None;
                    }
                    offer_state.finished = true;
                    offer_state.active = false;
                    offer_state.offer_id
                };
                state
                    .data_device_state
                    .dropped_dnd_offers
                    .remove(&finish_offer_id);
                state
                    .serializer
                    .writer()
                    .send(SendType::Object(Request::Data(
                        DataRequest::DestinationRequest(DataDestinationRequest::DnDFinish {
                            offer_id: finish_offer_id,
                        }),
                    )));
            },
            wl_data_offer::Request::SetActions {
                dnd_actions,
                preferred_action,
            } => {
                let Ok(dnd_actions) = dnd_actions.into_result() else {
                    offer.post_error(
                        wl_data_offer::Error::InvalidActionMask,
                        "Invalid dnd_actions mask.",
                    );
                    return None;
                };
                let Ok(preferred_action) = preferred_action.into_result() else {
                    offer.post_error(
                        wl_data_offer::Error::InvalidAction,
                        "Invalid preferred action.",
                    );
                    return None;
                };
                if !is_single_dnd_action(preferred_action) {
                    offer.post_error(
                        wl_data_offer::Error::InvalidAction,
                        "Invalid preferred action.",
                    );
                    return None;
                }
                let req = {
                    let mut offer_state = self.offer_state.lock().unwrap();
                    if !offer_state.active {
                        None
                    } else {
                        offer_state.destination_actions = dnd_actions;
                        offer_state.preferred_action = preferred_action;
                        offer_state.destination_actions_pending = true;
                        Some(DataDestinationRequest::DnDSetDestinationActions {
                            offer_id: offer_state.offer_id,
                            dnd_actions: dnd_actions.into(),
                            preferred_action: preferred_action.into(),
                        })
                    }
                };
                if let Some(req) = req {
                    state
                        .serializer
                        .writer()
                        .send(SendType::Object(Request::Data(
                            DataRequest::DestinationRequest(req),
                        )));
                }
            },
            _ => unreachable!(),
        }

        None
    }

    fn destroyed(
        self: Arc<Self>,
        _handle: &Handle,
        state: &mut WprsServerState,
        _client_id: ClientId,
        _object_id: ObjectId,
    ) {
        self.on_destroy(state);
    }
}

/// Pointer grab held while a proxied drag-and-drop session (remote-initiated or
/// host-initiated) is active on the host compositor. Blocks regular pointer
/// delivery without emitting unsolicited `wl_data_source.cancelled` on unset.
struct ProxyDndGrab {
    start_data: GrabStartData<WprsServerState>,
}

impl ProxyDndGrab {
    fn new(start_data: GrabStartData<WprsServerState>) -> Self {
        Self { start_data }
    }
}

impl PointerGrab<WprsServerState> for ProxyDndGrab {
    fn start_data(&self) -> &GrabStartData<WprsServerState> {
        &self.start_data
    }

    fn motion(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &MotionEvent,
    ) {
        handle.motion(data, None, event);
    }

    fn relative_motion(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        _event: &RelativeMotionEvent,
    ) {
    }

    fn button(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &ButtonEvent,
    ) {
    }

    fn axis(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _details: AxisFrame,
    ) {
    }

    fn frame(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
    ) {
    }

    fn gesture_swipe_begin(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GestureSwipeBeginEvent,
    ) {
    }

    fn gesture_swipe_update(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GestureSwipeUpdateEvent,
    ) {
    }

    fn gesture_swipe_end(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GestureSwipeEndEvent,
    ) {
    }

    fn gesture_pinch_begin(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GesturePinchBeginEvent,
    ) {
    }

    fn gesture_pinch_update(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GesturePinchUpdateEvent,
    ) {
    }

    fn gesture_pinch_end(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GesturePinchEndEvent,
    ) {
    }

    fn gesture_hold_begin(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GestureHoldBeginEvent,
    ) {
    }

    fn gesture_hold_end(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        _event: &GestureHoldEndEvent,
    ) {
    }

    fn unset(&mut self, data: &mut WprsServerState) {
        data.data_device_state.proxy_dnd_grab_active = false;
    }
}

/// Local pointer grab for internal same-client drags (`start_drag(NULL, ...)`),
/// which are restricted to surfaces of the originating client and emit
/// `wl_data_device.drop` + `leave` upon release.
struct InternalDndGrab {
    start_data: GrabStartData<WprsServerState>,
    origin: WlSurface,
    current_focus: Option<WlSurface>,
}

impl InternalDndGrab {
    fn new(start_data: GrabStartData<WprsServerState>, origin: WlSurface) -> Self {
        Self {
            start_data,
            origin,
            current_focus: None,
        }
    }
}

impl PointerGrab<WprsServerState> for InternalDndGrab {
    fn start_data(&self) -> &GrabStartData<WprsServerState> {
        &self.start_data
    }

    fn motion(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &MotionEvent,
    ) {
        handle.motion(data, None, event);

        let new_surface = focus.as_ref().map(|(s, _)| s);
        if new_surface != self.current_focus.as_ref()
            && let Some(old) = self.current_focus.take()
        {
            data.data_device_state
                .for_client_devices(&old.id(), WlDataDevice::leave);
        }

        if let Some((surface, surface_loc)) = focus
            && self.origin.id().same_client_as(&surface.id())
        {
            let (x, y) = (event.location - surface_loc).into();
            if self.current_focus.is_none() {
                data.data_device_state
                    .for_client_devices(&surface.id(), |device| {
                        device.enter(event.serial.into(), &surface, x, y, None);
                    });
                self.current_focus = Some(surface);
            } else {
                data.data_device_state
                    .for_client_devices(&surface.id(), |device| {
                        device.motion(event.time, x, y);
                    });
            }
        }
    }

    fn relative_motion(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &RelativeMotionEvent,
    ) {
        handle.relative_motion(data, focus, event);
    }

    fn button(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &ButtonEvent,
    ) {
        if handle.current_pressed().is_empty() {
            if let Some(surface) = self.current_focus.take() {
                data.data_device_state
                    .for_client_devices(&surface.id(), |device| {
                        device.drop();
                        device.leave();
                    });
            }
            handle.unset_grab(self, data, event.serial, event.time, true);
            handle.frame(data);
        }
    }

    fn axis(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        details: AxisFrame,
    ) {
        handle.axis(data, details);
    }

    fn frame(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
    ) {
        handle.frame(data);
    }

    fn gesture_swipe_begin(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureSwipeBeginEvent,
    ) {
        handle.gesture_swipe_begin(data, event);
    }

    fn gesture_swipe_update(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureSwipeUpdateEvent,
    ) {
        handle.gesture_swipe_update(data, event);
    }

    fn gesture_swipe_end(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureSwipeEndEvent,
    ) {
        handle.gesture_swipe_end(data, event);
    }

    fn gesture_pinch_begin(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GesturePinchBeginEvent,
    ) {
        handle.gesture_pinch_begin(data, event);
    }

    fn gesture_pinch_update(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GesturePinchUpdateEvent,
    ) {
        handle.gesture_pinch_update(data, event);
    }

    fn gesture_pinch_end(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GesturePinchEndEvent,
    ) {
        handle.gesture_pinch_end(data, event);
    }

    fn gesture_hold_begin(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureHoldBeginEvent,
    ) {
        handle.gesture_hold_begin(data, event);
    }

    fn gesture_hold_end(
        &mut self,
        data: &mut WprsServerState,
        handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureHoldEndEvent,
    ) {
        handle.gesture_hold_end(data, event);
    }

    fn unset(&mut self, data: &mut WprsServerState) {
        if let Some(surface) = self.current_focus.take() {
            data.data_device_state
                .for_client_devices(&surface.id(), WlDataDevice::leave);
        }
    }
}

pub struct WprsDataDeviceState {
    _manager_global: GlobalId,
    devices: Vec<Weak<WlDataDevice>>,
    clipboard_focus: Option<Client>,
    clipboard_selection: Option<ClipboardSelection>,
    next_source_id: u64,
    active_dnd_source: Option<ActiveDndSource>,
    active_incoming_dnd: Option<ActiveIncomingDnd>,
    dropped_dnd_offers: HashMap<u64, DroppedDndOffer>,
    proxy_dnd_grab_active: bool,
}

impl WprsDataDeviceState {
    pub fn new(dh: &DisplayHandle) -> Self {
        let manager_global = dh.create_global::<WprsServerState, WlDataDeviceManager, _>(3, ());
        Self {
            _manager_global: manager_global,
            devices: Vec::new(),
            clipboard_focus: None,
            clipboard_selection: None,
            next_source_id: 1,
            active_dnd_source: None,
            active_incoming_dnd: None,
            dropped_dnd_offers: HashMap::new(),
            proxy_dnd_grab_active: false,
        }
    }

    pub fn is_proxy_dnd_grab_active(&self) -> bool {
        self.proxy_dnd_grab_active
    }

    fn allocate_source_id(&mut self) -> u64 {
        let id = self.next_source_id;
        self.next_source_id += 1;
        id
    }

    fn for_client_devices(&mut self, id: &ObjectId, mut f: impl FnMut(&WlDataDevice)) {
        self.devices.retain(|weak| {
            let Ok(device) = weak.upgrade() else {
                return false;
            };
            if device.id().same_client_as(id) {
                f(&device);
            }
            true
        });
    }

    fn send_clipboard_to_device(
        dh: &DisplayHandle,
        device: &WlDataDevice,
        client_id: &ClientId,
        selection: Option<&ClipboardSelection>,
    ) {
        let Some(selection) = selection else {
            device.selection(None);
            return;
        };
        let Ok(offer_id) = dh.backend_handle().create_object::<WprsServerState>(
            client_id.clone(),
            WlDataOffer::interface(),
            device.version(),
            Arc::new(ClipboardOfferData {
                selection: selection.clone(),
            }),
        ) else {
            return;
        };
        let Ok(offer) = WlDataOffer::from_id(dh, offer_id) else {
            return;
        };
        device.data_offer(&offer);
        for mime_type in selection.mime_types() {
            offer.offer(mime_type);
        }
        device.selection(Some(&offer));
    }

    fn send_clipboard_to_focused_client(&mut self, dh: &DisplayHandle) {
        if matches!(
            &self.clipboard_selection,
            Some(ClipboardSelection::Client(source)) if !source_is_alive(source)
        ) {
            self.clipboard_selection = None;
        }
        let Some(focused_client) = self.clipboard_focus.clone() else {
            return;
        };
        let focused_id = focused_client.id();
        let selection = self.clipboard_selection.clone();
        self.devices.retain(|weak| {
            let Ok(device) = weak.upgrade() else {
                return false;
            };
            if dh
                .get_client(device.id())
                .is_ok_and(|c| c.id() == focused_id)
            {
                Self::send_clipboard_to_device(dh, &device, &focused_id, selection.as_ref());
            }
            true
        });
    }

    pub fn set_clipboard_focus(&mut self, dh: &DisplayHandle, client: Option<Client>) {
        if self.clipboard_focus != client {
            self.clipboard_focus = client;
            self.send_clipboard_to_focused_client(dh);
        }
    }

    pub fn set_host_clipboard_selection(&mut self, dh: &DisplayHandle, mime_types: Vec<String>) {
        if let Some(ClipboardSelection::Client(old)) = self.clipboard_selection.take()
            && source_is_alive(&old)
        {
            old.cancelled();
        }
        self.clipboard_selection = Some(ClipboardSelection::Host(mime_types));
        self.send_clipboard_to_focused_client(dh);
    }

    pub fn request_client_clipboard(&self, mime_type: String, fd: OwnedFd) -> Result<()> {
        match &self.clipboard_selection {
            Some(ClipboardSelection::Client(source)) => {
                if !source_is_alive(source) {
                    bail!("active clipboard data source is no longer alive");
                }
                if !source_contains_mime(source, &mime_type) {
                    bail!("requested clipboard mime type {mime_type:?} is not available");
                }
                source.send(mime_type, fd.as_fd());
                Ok(())
            },
            Some(ClipboardSelection::Host(_)) => bail!("current clipboard selection is host-owned"),
            None => bail!("no active clipboard selection"),
        }
    }

    pub fn request_client_dnd_send(
        &self,
        source_id: u64,
        mime_type: String,
        fd: OwnedFd,
    ) -> Result<()> {
        let Some(active) = &self.active_dnd_source else {
            bail!("no active DnD source for id {source_id}");
        };
        if active.id != source_id {
            bail!(
                "stale DnD send request for source {source_id} (active is {})",
                active.id
            );
        }
        if !source_is_alive(&active.source) {
            bail!("active DnD source {source_id} is no longer alive");
        }
        active.source.send(mime_type, fd.as_fd());
        Ok(())
    }

    fn offers_for_id(&self, offer_id: u64) -> Option<(&Arc<Mutex<DndOfferState>>, &[WlDataOffer])> {
        if let Some(active) = &self.active_incoming_dnd
            && active.offer_id == offer_id
        {
            return Some((&active.offer_state, &active.offers));
        }
        self.dropped_dnd_offers
            .get(&offer_id)
            .map(|dropped| (&dropped.offer_state, dropped.offers.as_slice()))
    }
}

impl WprsServerState {
    fn install_proxy_dnd_grab(&mut self, start_data: GrabStartData<Self>, serial: Serial) {
        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };
        self.data_device_state.proxy_dnd_grab_active = true;
        pointer.set_grab(self, ProxyDndGrab::new(start_data), serial, Focus::Clear);
    }

    fn release_pressed_buttons_and_unset_dnd_grab(&mut self) {
        if !self.data_device_state.proxy_dnd_grab_active {
            return;
        }
        let Some(pointer) = self.seat.get_pointer() else {
            self.data_device_state.proxy_dnd_grab_active = false;
            self.pressed_buttons.clear();
            return;
        };
        let time = self.start_time.elapsed().as_millis() as u32;
        for button in mem::take(&mut self.pressed_buttons) {
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
                location: pointer.current_location(),
                serial: SERIAL_COUNTER.next_serial(),
                time,
            },
        );
        pointer.unset_grab(self, SERIAL_COUNTER.next_serial(), time);
        self.data_device_state.proxy_dnd_grab_active = false;
    }

    fn leave_active_incoming_dnd(&mut self) {
        if let Some(active) = self.data_device_state.active_incoming_dnd.take() {
            active.offer_state.lock().unwrap().active = false;
            self.data_device_state
                .for_client_devices(&active.surface.id(), WlDataDevice::leave);
        }
    }

    pub(crate) fn on_dnd_surface_destroyed(&mut self, surface: &WlSurface) {
        if self
            .data_device_state
            .active_incoming_dnd
            .as_ref()
            .is_some_and(|active| active.surface == *surface)
            && let Some(active) = self.data_device_state.active_incoming_dnd.take()
        {
            active.offer_state.lock().unwrap().active = false;
        }
    }

    fn on_data_source_destroyed(&mut self, source: &WlDataSource) {
        if self
            .data_device_state
            .active_dnd_source
            .as_ref()
            .is_some_and(|active| active.source == *source)
            && let Some(active) = self.data_device_state.active_dnd_source.take()
        {
            self.serializer
                .writer()
                .send(SendType::Object(Request::Data(DataRequest::SourceRequest(
                    DataSourceRequest::CancelDrag(active.id),
                ))));
            if self.data_device_state.active_incoming_dnd.is_none() {
                self.release_pressed_buttons_and_unset_dnd_grab();
            }
        }

        if matches!(
            &self.data_device_state.clipboard_selection,
            Some(ClipboardSelection::Client(current)) if current == source
        ) {
            self.data_device_state.clipboard_selection = None;
            self.data_device_state
                .send_clipboard_to_focused_client(&self.dh);
        }
    }

    fn set_client_clipboard_selection(&mut self, new_source: Option<WlDataSource>) {
        if let Some(ClipboardSelection::Client(old)) = &self.data_device_state.clipboard_selection
            && new_source.as_ref() != Some(old)
            && source_is_alive(old)
        {
            old.cancelled();
        }
        if let Some(source) = &new_source {
            let metadata = SourceMetadata::from_mime_types(source_metadata(source).mime_types);
            self.serializer
                .writer()
                .send(SendType::Object(Request::Data(DataRequest::SourceRequest(
                    DataSourceRequest::SetSelection(DataSource::Selection, metadata),
                ))));
        }
        self.data_device_state.clipboard_selection = new_source.map(ClipboardSelection::Client);
        self.data_device_state
            .send_clipboard_to_focused_client(&self.dh);
    }

    fn handle_start_drag(
        &mut self,
        device: &WlDataDevice,
        source: Option<WlDataSource>,
        origin: WlSurface,
        icon: Option<WlSurface>,
        serial: Serial,
    ) {
        let Some(pointer) = self.seat.get_pointer() else {
            if let Some(source) = source {
                source.cancelled();
            }
            return;
        };

        if !pointer.has_grab(serial) {
            debug!(?serial, "denying drag from client without implicit grab");
            if let Some(source) = source {
                source.cancelled();
            }
            return;
        }

        if let Some(icon) = &icon
            && compositor::give_role(icon, DND_ICON_ROLE).is_err()
        {
            device.post_error(
                wl_data_device::Error::Role,
                "Given surface already has another role.",
            );
            return;
        }

        let Some(source) = source else {
            let start_data = pointer.grab_start_data().unwrap();
            pointer.set_grab(
                self,
                InternalDndGrab::new(start_data, origin),
                serial,
                Focus::Clear,
            );
            return;
        };

        if !self.serializer.other_end_connected() {
            debug!("denying drag while disconnected from wprsc");
            source.cancelled();
            return;
        }

        let Some(host_serial) = self.serial_map.get(serial) else {
            warn!("denying drag with unknown serial {serial:?}");
            source.cancelled();
            return;
        };

        let Ok(origin_surface) = ClientSurface::new(&origin).log(loc!()) else {
            source.cancelled();
            return;
        };

        let icon_surface = icon.as_ref().and_then(|surface| {
            let client_surface = ClientSurface::new(surface).log(loc!()).ok()?;
            self.insert_surface(surface).log(loc!()).ok()?;
            self.sync_surface_if_unsynced(surface);
            Some(client_surface)
        });

        let metadata = source_metadata(&source);
        if let Some(old) = self.data_device_state.active_dnd_source.take() {
            if old.source != source && source_is_alive(&old.source) {
                old.source.cancelled();
            }
            self.serializer
                .writer()
                .send(SendType::Object(Request::Data(DataRequest::SourceRequest(
                    DataSourceRequest::CancelDrag(old.id),
                ))));
        }

        let source_id = self.data_device_state.allocate_source_id();
        self.data_device_state.active_dnd_source = Some(ActiveDndSource {
            id: source_id,
            source,
            dropped: false,
        });

        let start_data = pointer.grab_start_data().unwrap();
        self.install_proxy_dnd_grab(start_data, serial);

        debug!("START DRAG: id={source_id}, metadata={metadata:?}");
        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(DataRequest::SourceRequest(
                DataSourceRequest::StartDrag(StartDrag {
                    source_id,
                    source: metadata,
                    origin: origin_surface,
                    icon: icon_surface,
                    serial: host_serial,
                }),
            ))));
    }

    pub(crate) fn handle_dnd_enter(
        &mut self,
        drag_enter: DragEnter,
        surface: WlSurface,
    ) -> Result<()> {
        let source_actions: DndAction = drag_enter
            .source_actions
            .try_into()
            .map_err(|()| anyhow!("invalid dnd source actions"))
            .location(loc!())?;
        let selected_action: DndAction = drag_enter
            .selected_action
            .try_into()
            .map_err(|()| anyhow!("invalid dnd selected action"))
            .location(loc!())?;

        // Never send two overlapping `wl_data_device.enter` events without an
        // intervening `leave`.
        self.leave_active_incoming_dnd();

        let serial = self.serial_map.insert(drag_enter.serial);
        if !self.data_device_state.proxy_dnd_grab_active {
            let start_data = GrabStartData {
                focus: Some((surface.clone(), (0.0, 0.0).into())),
                button: 0,
                location: drag_enter.loc.into(),
            };
            self.install_proxy_dnd_grab(start_data, serial);
        }

        let Ok(client) = self.dh.get_client(surface.id()) else {
            return Ok(());
        };
        let offer_state = Arc::new(Mutex::new(DndOfferState {
            offer_id: drag_enter.offer_id,
            mime_types: drag_enter.mime_types.clone(),
            source_actions,
            host_selected_action: selected_action,
            destination_actions: DndAction::None,
            preferred_action: DndAction::None,
            destination_actions_pending: false,
            active: true,
            dropped: false,
            accepted: false,
            finished: false,
            live_offer_count: 0,
        }));

        let mut offers = Vec::new();
        let dh = self.dh.clone();
        self.data_device_state
            .for_client_devices(&surface.id(), |device| {
                let Ok(offer_obj_id) = dh.backend_handle().create_object::<Self>(
                    client.id(),
                    WlDataOffer::interface(),
                    device.version(),
                    Arc::new(DndOfferObjectData {
                        offer_state: Arc::clone(&offer_state),
                        destroyed: AtomicBool::new(false),
                    }),
                ) else {
                    return;
                };
                let Ok(offer) = WlDataOffer::from_id(&dh, offer_obj_id) else {
                    return;
                };

                device.data_offer(&offer);
                for mime_type in &drag_enter.mime_types {
                    offer.offer(mime_type.clone());
                }
                if offer.version() >= wl_data_offer::EVT_SOURCE_ACTIONS_SINCE {
                    offer.source_actions(source_actions);
                }
                device.enter(
                    serial.into(),
                    &surface,
                    drag_enter.loc.x,
                    drag_enter.loc.y,
                    Some(&offer),
                );
                if !selected_action.is_empty() && offer.version() >= wl_data_offer::EVT_ACTION_SINCE
                {
                    offer.action(selected_action);
                }
                offers.push(offer);
            });

        offer_state.lock().unwrap().live_offer_count = offers.len();
        self.data_device_state.active_incoming_dnd = Some(ActiveIncomingDnd {
            offer_id: drag_enter.offer_id,
            surface,
            offer_state,
            offers,
        });
        Ok(())
    }

    pub(crate) fn handle_dnd_motion(&mut self, offer_id: u64, x: f64, y: f64) {
        let Some(active) = &self.data_device_state.active_incoming_dnd else {
            return;
        };
        if active.offer_id != offer_id {
            return;
        }
        let surface_id = active.surface.id();
        let time = self.start_time.elapsed().as_millis() as u32;
        self.data_device_state
            .for_client_devices(&surface_id, |device| device.motion(time, x, y));
    }

    pub(crate) fn handle_dnd_leave(&mut self, offer_id: u64) {
        let matches_active = self
            .data_device_state
            .active_incoming_dnd
            .as_ref()
            .is_some_and(|active| active.offer_id == offer_id);
        if !matches_active {
            return;
        }
        self.leave_active_incoming_dnd();
        // Only release the pointer grab if no remote-initiated source drag is
        // still in flight (e.g., dragging off a remote window onto a host app).
        let remote_drag_grab_ongoing = self
            .data_device_state
            .active_dnd_source
            .as_ref()
            .is_some_and(|src| !src.dropped);
        if !remote_drag_grab_ongoing {
            self.release_pressed_buttons_and_unset_dnd_grab();
        }
    }

    pub(crate) fn handle_dnd_drop(&mut self, offer_id: u64) {
        let matches_active = self
            .data_device_state
            .active_incoming_dnd
            .as_ref()
            .is_some_and(|active| active.offer_id == offer_id);
        if !matches_active {
            return;
        }
        let active = self.data_device_state.active_incoming_dnd.take().unwrap();
        active.offer_state.lock().unwrap().dropped = true;

        self.data_device_state
            .for_client_devices(&active.surface.id(), |device| {
                device.drop();
                device.leave();
            });

        if !active.offers.is_empty() {
            self.data_device_state.dropped_dnd_offers.insert(
                offer_id,
                DroppedDndOffer {
                    offer_state: active.offer_state,
                    offers: active.offers,
                },
            );
        }
        if let Some(active_source) = &mut self.data_device_state.active_dnd_source {
            active_source.dropped = true;
        }
        self.release_pressed_buttons_and_unset_dnd_grab();
    }

    pub(crate) fn handle_dnd_offer_source_actions(
        &mut self,
        offer_id: u64,
        actions: u32,
    ) -> Result<()> {
        let actions: DndAction = actions
            .try_into()
            .map_err(|()| anyhow!("invalid dnd source actions"))
            .location(loc!())?;
        let Some((offer_state, offers)) = self.data_device_state.offers_for_id(offer_id) else {
            return Ok(());
        };
        offer_state.lock().unwrap().source_actions = actions;
        for offer in offers {
            if offer.version() >= wl_data_offer::EVT_SOURCE_ACTIONS_SINCE {
                offer.source_actions(actions);
            }
        }
        Ok(())
    }

    pub(crate) fn handle_dnd_offer_selected_action(
        &mut self,
        offer_id: u64,
        action: u32,
    ) -> Result<()> {
        let action: DndAction = action
            .try_into()
            .map_err(|()| anyhow!("invalid dnd selected action"))
            .location(loc!())?;
        let Some((offer_state, offers)) = self.data_device_state.offers_for_id(offer_id) else {
            return Ok(());
        };
        {
            let mut offer_state = offer_state.lock().unwrap();
            offer_state.host_selected_action = action;
            offer_state.destination_actions_pending = false;
        }
        for offer in offers {
            if offer.version() >= wl_data_offer::EVT_ACTION_SINCE {
                offer.action(action);
            }
        }
        Ok(())
    }

    pub(crate) fn handle_dnd_source_target(&self, source_id: u64, mime_type: Option<String>) {
        if let Some(active) = &self.data_device_state.active_dnd_source
            && active.id == source_id
            && source_is_alive(&active.source)
        {
            active.source.target(mime_type);
        }
    }

    pub(crate) fn handle_dnd_source_action(&self, source_id: u64, action: u32) -> Result<()> {
        let action: DndAction = action
            .try_into()
            .map_err(|()| anyhow!("invalid dnd source action"))
            .location(loc!())?;
        if let Some(active) = &self.data_device_state.active_dnd_source
            && active.id == source_id
            && source_is_alive(&active.source)
            && active.source.version() >= wl_data_source::EVT_ACTION_SINCE
        {
            active.source.action(action);
        }
        Ok(())
    }

    pub(crate) fn handle_dnd_source_drop_performed(&mut self, source_id: u64) {
        if let Some(active) = &mut self.data_device_state.active_dnd_source
            && active.id == source_id
        {
            active.dropped = true;
            if source_is_alive(&active.source)
                && active.source.version() >= wl_data_source::EVT_DND_DROP_PERFORMED_SINCE
            {
                active.source.dnd_drop_performed();
            }
            if self.data_device_state.active_incoming_dnd.is_none() {
                self.release_pressed_buttons_and_unset_dnd_grab();
            }
        }
    }

    pub(crate) fn handle_dnd_source_finished_or_cancelled(
        &mut self,
        source_id: u64,
        finished: bool,
    ) {
        if self
            .data_device_state
            .active_dnd_source
            .as_ref()
            .is_some_and(|active| active.id == source_id)
            && let Some(active) = self.data_device_state.active_dnd_source.take()
        {
            if source_is_alive(&active.source) {
                if finished {
                    if active.source.version() >= wl_data_source::EVT_DND_FINISHED_SINCE {
                        active.source.dnd_finished();
                    }
                } else {
                    active.source.cancelled();
                }
            }
            if self.data_device_state.active_incoming_dnd.is_none() {
                self.release_pressed_buttons_and_unset_dnd_grab();
            }
        }
    }

    pub(crate) fn reset_dnd_state(&mut self) {
        if let Some(active) = self.data_device_state.active_dnd_source.take()
            && source_is_alive(&active.source)
        {
            active.source.cancelled();
        }
        self.leave_active_incoming_dnd();
        for (_, dropped) in self.data_device_state.dropped_dnd_offers.drain() {
            dropped.offer_state.lock().unwrap().active = false;
        }
        self.release_pressed_buttons_and_unset_dnd_grab();
        self.pending_receive_pipes.clear();
    }
}

impl GlobalDispatch<WlDataDeviceManager, ()> for WprsServerState {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<WlDataDeviceManager>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<WlDataDeviceManager, ()> for WprsServerState {
    fn request(
        state: &mut Self,
        client: &Client,
        _resource: &WlDataDeviceManager,
        request: wl_data_device_manager::Request,
        _data: &(),
        dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device_manager::Request::CreateDataSource { id } => {
                data_init.init(id, DataSourceUserData::new());
            },
            wl_data_device_manager::Request::GetDataDevice { id, seat: wl_seat } => {
                let device = data_init.init(
                    id,
                    DataDeviceUserData {
                        wl_seat: wl_seat.clone(),
                    },
                );
                if Seat::<Self>::from_resource(&wl_seat).is_some_and(|s| s == state.seat) {
                    state.data_device_state.devices.push(device.downgrade());
                    if state.data_device_state.clipboard_focus.as_ref() == Some(client) {
                        if matches!(
                            &state.data_device_state.clipboard_selection,
                            Some(ClipboardSelection::Client(source)) if !source_is_alive(source)
                        ) {
                            state.data_device_state.clipboard_selection = None;
                        }
                        WprsDataDeviceState::send_clipboard_to_device(
                            dh,
                            &device,
                            &client.id(),
                            state.data_device_state.clipboard_selection.as_ref(),
                        );
                    }
                } else {
                    error!(
                        ?client,
                        ?device,
                        "Unmanaged seat given to a wl_data_device."
                    );
                }
            },
            _ => unreachable!(),
        }
    }
}

impl Dispatch<WlDataSource, DataSourceUserData> for WprsServerState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &WlDataSource,
        request: wl_data_source::Request,
        data: &DataSourceUserData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_source::Request::Offer { mime_type } => {
                data.inner.lock().unwrap().mime_types.push(mime_type);
            },
            wl_data_source::Request::SetActions { dnd_actions } => match dnd_actions {
                WEnum::Value(dnd_actions) => {
                    data.inner.lock().unwrap().dnd_actions = dnd_actions;
                },
                WEnum::Unknown(action) => {
                    error!("Unknown dnd_action: {action:?}");
                },
            },
            wl_data_source::Request::Destroy => {
                data.alive.store(false, Ordering::Release);
                state.on_data_source_destroyed(resource);
            },
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &WlDataSource,
        data: &DataSourceUserData,
    ) {
        data.alive.store(false, Ordering::Release);
        state.on_data_source_destroyed(resource);
    }
}

impl Dispatch<WlDataDevice, DataDeviceUserData> for WprsServerState {
    fn request(
        state: &mut Self,
        client: &Client,
        resource: &WlDataDevice,
        request: wl_data_device::Request,
        data: &DataDeviceUserData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        if Seat::<Self>::from_resource(&data.wl_seat).is_none_or(|s| s != state.seat) {
            return;
        }

        match request {
            wl_data_device::Request::StartDrag {
                source,
                origin,
                icon,
                serial,
            } => {
                state.handle_start_drag(resource, source, origin, icon, Serial::from(serial));
            },
            wl_data_device::Request::SetSelection { source, .. } => {
                let has_focus = state
                    .seat
                    .get_keyboard()
                    .is_some_and(|kbd| kbd.client_of_object_has_focus(&resource.id()));
                if !has_focus {
                    debug!(?client, "denying setting selection by a non-focused client");
                    return;
                }
                state.set_client_clipboard_selection(source);
            },
            wl_data_device::Request::Release => {
                state
                    .data_device_state
                    .devices
                    .retain(|weak| weak.id() != resource.id());
            },
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &WlDataDevice,
        _data: &DataDeviceUserData,
    ) {
        state
            .data_device_state
            .devices
            .retain(|weak| weak.id() != resource.id());
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use crossbeam_channel::Receiver;
    use nix::fcntl::OFlag;
    use nix::unistd;
    use smithay::reexports::calloop::EventLoop;
    use smithay::reexports::wayland_server::Display;
    use smithay_client_toolkit::reexports::client::Connection;
    use smithay_client_toolkit::reexports::client::Dispatch as ClientDispatch;
    use smithay_client_toolkit::reexports::client::EventQueue;
    use smithay_client_toolkit::reexports::client::Proxy;
    use smithay_client_toolkit::reexports::client::QueueHandle;
    use smithay_client_toolkit::reexports::client::WEnum as ClientWEnum;
    use smithay_client_toolkit::reexports::client::event_created_child;
    use smithay_client_toolkit::reexports::client::protocol::wl_callback::WlCallback;
    use smithay_client_toolkit::reexports::client::protocol::wl_compositor::WlCompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_device as client_data_device;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_device::WlDataDevice as ClientWlDataDevice;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_device_manager::DndAction as ClientDndAction;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_device_manager::WlDataDeviceManager as ClientWlDataDeviceManager;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_offer as client_data_offer;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_offer::WlDataOffer as ClientWlDataOffer;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_source as client_data_source;
    use smithay_client_toolkit::reexports::client::protocol::wl_data_source::WlDataSource as ClientWlDataSource;
    use smithay_client_toolkit::reexports::client::protocol::wl_keyboard::WlKeyboard as ClientWlKeyboard;
    use smithay_client_toolkit::reexports::client::protocol::wl_pointer as client_pointer;
    use smithay_client_toolkit::reexports::client::protocol::wl_pointer::WlPointer as ClientWlPointer;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry::WlRegistry;
    use smithay_client_toolkit::reexports::client::protocol::wl_seat::WlSeat as ClientWlSeat;
    use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as ClientWlSurface;

    use super::*;
    use crate::serialization::Event;
    use crate::serialization::RecvType;
    use crate::serialization::Serializer;
    use crate::serialization::wayland::DataDestinationEvent;
    use crate::serialization::wayland::DataEvent;
    use crate::serialization::wayland::DataSourceEvent;
    use crate::serialization::wayland::DataToTransfer;
    use crate::serialization::wayland::KeyboardEvent;
    use crate::serialization::wayland::PointerEvent;
    use crate::serialization::wayland::PointerEventKind;
    use crate::serialization::wayland::WlSurfaceId;
    use crate::server::smithay_handlers::ClientState;

    #[derive(Debug, PartialEq)]
    enum RecordedEvent {
        PointerEnter(u32),
        PointerLeave(u32),
        PointerMotion(f64, f64),
        PointerButton {
            serial: u32,
            button: u32,
            pressed: bool,
        },
        SourceTarget(Option<String>),
        SourceAction(ClientDndAction),
        SourceDropPerformed,
        SourceFinished,
        SourceCancelled,
        SourceSend(String),
        OfferMime(String),
        OfferSourceActions(ClientDndAction),
        OfferAction(ClientDndAction),
        DeviceEnter {
            serial: u32,
            has_offer: bool,
            x: f64,
            y: f64,
        },
        DeviceMotion(f64, f64),
        DeviceDrop,
        DeviceLeave,
        DeviceSelection(bool),
    }

    #[derive(Default)]
    struct WireClientData {
        compositor: Option<WlCompositor>,
        seat: Option<ClientWlSeat>,
        data_device_manager: Option<ClientWlDataDeviceManager>,
        events: Vec<RecordedEvent>,
        last_offer: Option<ClientWlDataOffer>,
        send_fds: Vec<(String, OwnedFd)>,
    }

    impl ClientDispatch<WlRegistry, ()> for WireClientData {
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
                        state.compositor = Some(registry.bind(name, version.min(5), qh, ()));
                    },
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(8), qh, ()));
                    },
                    "wl_data_device_manager" => {
                        state.data_device_manager =
                            Some(registry.bind(name, version.min(3), qh, ()));
                    },
                    _ => {},
                }
            }
        }
    }

    impl ClientDispatch<WlCallback, Arc<AtomicBool>> for WireClientData {
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

    impl ClientDispatch<WlCompositor, ()> for WireClientData {
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

    impl ClientDispatch<ClientWlSurface, ()> for WireClientData {
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

    impl ClientDispatch<ClientWlSeat, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &ClientWlSeat,
            _event: <ClientWlSeat as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl ClientDispatch<ClientWlKeyboard, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &ClientWlKeyboard,
            _event: <ClientWlKeyboard as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl ClientDispatch<ClientWlPointer, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &ClientWlPointer,
            event: client_pointer::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                client_pointer::Event::Enter { serial, .. } => {
                    state.events.push(RecordedEvent::PointerEnter(serial));
                },
                client_pointer::Event::Leave { serial, .. } => {
                    state.events.push(RecordedEvent::PointerLeave(serial));
                },
                client_pointer::Event::Motion {
                    surface_x,
                    surface_y,
                    ..
                } => {
                    state
                        .events
                        .push(RecordedEvent::PointerMotion(surface_x, surface_y));
                },
                client_pointer::Event::Button {
                    serial,
                    button,
                    state: ClientWEnum::Value(btn_state),
                    ..
                } => {
                    state.events.push(RecordedEvent::PointerButton {
                        serial,
                        button,
                        pressed: btn_state == client_pointer::ButtonState::Pressed,
                    });
                },
                _ => {},
            }
        }
    }

    impl ClientDispatch<ClientWlDataDeviceManager, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &ClientWlDataDeviceManager,
            _event: <ClientWlDataDeviceManager as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl ClientDispatch<ClientWlDataSource, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &ClientWlDataSource,
            event: client_data_source::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                client_data_source::Event::Target { mime_type } => {
                    state.events.push(RecordedEvent::SourceTarget(mime_type));
                },
                client_data_source::Event::Send { mime_type, fd } => {
                    state
                        .events
                        .push(RecordedEvent::SourceSend(mime_type.clone()));
                    state.send_fds.push((mime_type, fd));
                },
                client_data_source::Event::Cancelled => {
                    state.events.push(RecordedEvent::SourceCancelled);
                },
                client_data_source::Event::DndDropPerformed => {
                    state.events.push(RecordedEvent::SourceDropPerformed);
                },
                client_data_source::Event::DndFinished => {
                    state.events.push(RecordedEvent::SourceFinished);
                },
                client_data_source::Event::Action {
                    dnd_action: ClientWEnum::Value(action),
                } => {
                    state.events.push(RecordedEvent::SourceAction(action));
                },
                _ => {},
            }
        }
    }

    impl ClientDispatch<ClientWlDataOffer, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &ClientWlDataOffer,
            event: client_data_offer::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                client_data_offer::Event::Offer { mime_type } => {
                    state.events.push(RecordedEvent::OfferMime(mime_type));
                },
                client_data_offer::Event::SourceActions {
                    source_actions: ClientWEnum::Value(actions),
                } => {
                    state
                        .events
                        .push(RecordedEvent::OfferSourceActions(actions));
                },
                client_data_offer::Event::Action {
                    dnd_action: ClientWEnum::Value(action),
                } => {
                    state.events.push(RecordedEvent::OfferAction(action));
                },
                _ => {},
            }
        }
    }

    impl ClientDispatch<ClientWlDataDevice, ()> for WireClientData {
        event_created_child!(WireClientData, ClientWlDataDevice, [
            client_data_device::EVT_DATA_OFFER_OPCODE => (ClientWlDataOffer, ()),
        ]);

        fn event(
            state: &mut Self,
            _proxy: &ClientWlDataDevice,
            event: client_data_device::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                client_data_device::Event::DataOffer { id } => {
                    state.last_offer = Some(id);
                },
                client_data_device::Event::Enter {
                    serial, x, y, id, ..
                } => {
                    let has_offer = id.is_some();
                    if let Some(offer) = id {
                        state.last_offer = Some(offer);
                    }
                    state.events.push(RecordedEvent::DeviceEnter {
                        serial,
                        has_offer,
                        x,
                        y,
                    });
                },
                client_data_device::Event::Leave => {
                    state.events.push(RecordedEvent::DeviceLeave);
                },
                client_data_device::Event::Motion { x, y, .. } => {
                    state.events.push(RecordedEvent::DeviceMotion(x, y));
                },
                client_data_device::Event::Drop => {
                    state.events.push(RecordedEvent::DeviceDrop);
                },
                client_data_device::Event::Selection { id } => {
                    let has_offer = id.is_some();
                    if let Some(offer) = id {
                        state.last_offer = Some(offer);
                    }
                    state.events.push(RecordedEvent::DeviceSelection(has_offer));
                },
                _ => {},
            }
        }
    }

    struct WireClient {
        conn: Connection,
        event_queue: EventQueue<WireClientData>,
        qh: QueueHandle<WireClientData>,
        data: WireClientData,
        _pointer: ClientWlPointer,
        _keyboard: ClientWlKeyboard,
        data_device: ClientWlDataDevice,
        surface: ClientWlSurface,
        surface_id: WlSurfaceId,
    }

    struct DndHarness {
        display: Display<WprsServerState>,
        _event_loop: EventLoop<'static, WprsServerState>,
        state: WprsServerState,
        rx: Option<Receiver<SendType<Request>>>,
        conn_id: Option<u64>,
    }

    impl DndHarness {
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
            state
                .seat
                .add_keyboard(Default::default(), 200, 200)
                .unwrap();
            state.seat.add_pointer();
            let mut harness = Self {
                display,
                _event_loop: event_loop,
                state,
                rx: None,
                conn_id: None,
            };
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

        fn add_client(&mut self) -> WireClient {
            let (server_sock, client_sock) = UnixStream::pair().unwrap();
            self.display
                .handle()
                .insert_client(
                    server_sock,
                    Arc::new(ClientState::new(self.state.serializer.writer())),
                )
                .unwrap();

            let conn = Connection::from_socket(client_sock).unwrap();
            let mut event_queue: EventQueue<WireClientData> = conn.new_event_queue();
            let qh = event_queue.handle();
            let _registry = conn.display().get_registry(&qh, ());
            let mut data = WireClientData::default();
            Self::roundtrip_parts(
                &mut self.display,
                &mut self.state,
                &conn,
                &mut event_queue,
                &qh,
                &mut data,
            );

            let seat = data.seat.clone().unwrap();
            let pointer = seat.get_pointer(&qh, ());
            let keyboard = seat.get_keyboard(&qh, ());
            let data_device =
                data.data_device_manager
                    .as_ref()
                    .unwrap()
                    .get_data_device(&seat, &qh, ());
            let existing_surfaces: std::collections::HashSet<WlSurfaceId> =
                self.state.object_map.keys().copied().collect();
            let surface = data.compositor.as_ref().unwrap().create_surface(&qh, ());
            surface.commit();
            Self::roundtrip_parts(
                &mut self.display,
                &mut self.state,
                &conn,
                &mut event_queue,
                &qh,
                &mut data,
            );
            data.events.clear();
            let surface_id = *self
                .state
                .object_map
                .keys()
                .find(|sid| !existing_surfaces.contains(*sid))
                .expect("new client surface must be registered in object_map");

            WireClient {
                conn,
                event_queue,
                qh,
                data,
                _pointer: pointer,
                _keyboard: keyboard,
                data_device,
                surface,
                surface_id,
            }
        }

        fn roundtrip_parts(
            display: &mut Display<WprsServerState>,
            state: &mut WprsServerState,
            conn: &Connection,
            event_queue: &mut EventQueue<WireClientData>,
            qh: &QueueHandle<WireClientData>,
            data: &mut WireClientData,
        ) {
            let done = Arc::new(AtomicBool::new(false));
            let _cb = conn.display().sync(qh, Arc::clone(&done));
            conn.flush().unwrap();
            display.dispatch_clients(state).unwrap();
            display.flush_clients().unwrap();
            while !done.load(Ordering::Acquire) {
                event_queue.blocking_dispatch(data).unwrap();
            }
        }

        fn roundtrip(&mut self, client: &mut WireClient) {
            Self::roundtrip_parts(
                &mut self.display,
                &mut self.state,
                &client.conn,
                &mut client.event_queue,
                &client.qh,
                &mut client.data,
            );
        }

        fn send_event(&mut self, event: Event) {
            let conn_id = self.conn_id.expect("wprsc not connected");
            self.state.handle_event(conn_id, RecvType::Object(event));
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

        fn recv_request_timeout(&self, timeout: Duration) -> Request {
            let rx = self.rx.as_ref().expect("wprsc not connected");
            loop {
                match rx
                    .recv_timeout(timeout)
                    .expect("timed out waiting for request")
                {
                    SendType::Object(req) => return req,
                    SendType::RawBuffer(_) => {},
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::verbose_file_reads)]
    fn test_remote_to_remote_dnd_full_lifecycle_and_concurrent_transfer() {
        let mut h = DndHarness::new(true);
        let mut src_client = h.add_client();
        let mut dst_client = h.add_client();
        let src_surface_id = src_client.surface_id;
        let dst_surface_id = dst_client.surface_id;

        // 1. Pointer enters and presses BTN_LEFT on source surface.
        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id: src_surface_id,
                position: (10.0, 20.0).into(),
                kind: PointerEventKind::Enter { serial: 100 },
            },
            PointerEvent {
                surface_id: src_surface_id,
                position: (10.0, 20.0).into(),
                kind: PointerEventKind::Press {
                    button: 0x110,
                    serial: 101,
                },
            },
        ]));
        h.roundtrip(&mut src_client);
        let press_serial = src_client
            .data
            .events
            .iter()
            .find_map(|e| match *e {
                RecordedEvent::PointerButton {
                    serial,
                    button: 0x110,
                    pressed: true,
                } => Some(serial),
                _ => None,
            })
            .expect("source client should receive press serial");
        src_client.data.events.clear();
        h.drain_requests();

        // 2. Source client creates data source + drag icon and starts drag.
        let source = src_client
            .data
            .data_device_manager
            .as_ref()
            .unwrap()
            .create_data_source(&src_client.qh, ());
        source.offer("text/plain".to_string());
        source.offer("text/uri-list".to_string());
        source.set_actions(ClientDndAction::Copy | ClientDndAction::Move);
        let icon = src_client
            .data
            .compositor
            .as_ref()
            .unwrap()
            .create_surface(&src_client.qh, ());
        src_client.data_device.start_drag(
            Some(&source),
            &src_client.surface,
            Some(&icon),
            press_serial,
        );
        h.roundtrip(&mut src_client);

        assert!(h.state.data_device_state.is_proxy_dnd_grab_active());
        assert!(
            src_client
                .data
                .events
                .iter()
                .any(|e| matches!(e, RecordedEvent::PointerLeave(_))),
            "starting drag grab must clear pointer focus with wl_pointer.leave"
        );
        src_client.data.events.clear();
        let reqs = h.drain_requests();
        let start_drag = reqs
            .into_iter()
            .find_map(|r| match r {
                Request::Data(DataRequest::SourceRequest(DataSourceRequest::StartDrag(sd))) => {
                    Some(sd)
                },
                _ => None,
            })
            .expect("wprsd must emit StartDrag");
        assert_eq!(start_drag.serial, 101);
        assert_eq!(start_drag.origin.surface, src_surface_id);
        assert!(start_drag.icon.is_some());
        assert_eq!(
            start_drag.source.mime_types,
            vec!["text/plain".to_string(), "text/uri-list".to_string()]
        );
        assert_eq!(
            start_drag.source.dnd_actions,
            u32::from(DndAction::Copy | DndAction::Move)
        );
        let source_id = start_drag.source_id;

        // Host sends wl_pointer.leave once it grabs the drag; wprsd must keep
        // pressed_buttons intact while ProxyDndGrab is active.
        h.send_event(Event::PointerFrame(vec![PointerEvent {
            surface_id: src_surface_id,
            position: (10.0, 20.0).into(),
            kind: PointerEventKind::Leave { serial: 102 },
        }]));
        assert!(h.state.pressed_buttons.contains(&0x110));

        // 3. Host enters destination surface -> destination gets offer + enter.
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 50,
                serial: 200,
                surface: dst_surface_id,
                loc: (25.5, 40.0).into(),
                source_actions: (DndAction::Copy | DndAction::Move).into(),
                selected_action: DndAction::None.into(),
                mime_types: vec!["text/plain".to_string(), "text/uri-list".to_string()],
            }),
        )));
        h.roundtrip(&mut dst_client);

        let enter_serial = dst_client
            .data
            .events
            .iter()
            .find_map(|e| match *e {
                RecordedEvent::DeviceEnter {
                    serial,
                    has_offer: true,
                    x,
                    y,
                } => {
                    assert_eq!((x, y), (25.5, 40.0));
                    Some(serial)
                },
                _ => None,
            })
            .expect("dst should receive DeviceEnter");
        assert_ne!(enter_serial, 0);
        assert!(
            dst_client
                .data
                .events
                .contains(&RecordedEvent::OfferMime("text/plain".to_string()))
        );
        assert!(
            dst_client
                .data
                .events
                .contains(&RecordedEvent::OfferMime("text/uri-list".to_string()))
        );
        assert!(
            dst_client
                .data
                .events
                .contains(&RecordedEvent::OfferSourceActions(
                    ClientDndAction::Copy | ClientDndAction::Move
                ))
        );
        dst_client.data.events.clear();

        // 4. Destination accepts MIME + sets actions -> forwarded to host.
        let offer = dst_client.data.last_offer.clone().unwrap();
        offer.accept(enter_serial, Some("text/plain".to_string()));
        offer.set_actions(
            ClientDndAction::Copy | ClientDndAction::Move,
            ClientDndAction::Move,
        );
        h.roundtrip(&mut dst_client);
        let dst_reqs = h.drain_requests();
        assert_eq!(
            dst_reqs,
            vec![
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::DnDAcceptMimeType {
                        offer_id: 50,
                        mime_type: Some("text/plain".to_string()),
                    },
                )),
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::DnDSetDestinationActions {
                        offer_id: 50,
                        dnd_actions: (DndAction::Copy | DndAction::Move).into(),
                        preferred_action: DndAction::Move.into(),
                    },
                )),
            ]
        );

        // 5. Host echoes target + action to source and action + motion to dst.
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDMimeTypeAcceptedByDestination {
                source_id,
                mime_type: Some("text/plain".to_string()),
            },
        )));
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDActionSelected {
                source_id,
                action: DndAction::Move.into(),
            },
        )));
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDActionSelected {
                offer_id: 50,
                action: DndAction::Move.into(),
            },
        )));
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDMotion {
                offer_id: 50,
                loc: (30.0, 45.0).into(),
            },
        )));
        h.roundtrip(&mut src_client);
        h.roundtrip(&mut dst_client);

        assert_eq!(
            src_client.data.events,
            vec![
                RecordedEvent::SourceTarget(Some("text/plain".to_string())),
                RecordedEvent::SourceAction(ClientDndAction::Move),
            ]
        );
        assert_eq!(
            dst_client.data.events,
            vec![
                RecordedEvent::OfferAction(ClientDndAction::Move),
                RecordedEvent::DeviceMotion(30.0, 45.0),
            ]
        );
        src_client.data.events.clear();
        dst_client.data.events.clear();

        // 6. Drop occurs -> both sides notified, ProxyDndGrab unset so normal
        // pointer input works immediately afterwards.
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDDrop { offer_id: 50 },
        )));
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDDropPerformed { source_id },
        )));
        h.roundtrip(&mut src_client);
        h.roundtrip(&mut dst_client);

        assert!(!h.state.data_device_state.is_proxy_dnd_grab_active());
        assert_eq!(
            dst_client.data.events,
            vec![RecordedEvent::DeviceDrop, RecordedEvent::DeviceLeave]
        );
        assert_eq!(
            src_client.data.events,
            vec![RecordedEvent::SourceDropPerformed]
        );
        src_client.data.events.clear();
        dst_client.data.events.clear();

        // Normal pointer input resumes without being swallowed by a stuck grab.
        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id: dst_surface_id,
                position: (5.0, 6.0).into(),
                kind: PointerEventKind::Enter { serial: 300 },
            },
            PointerEvent {
                surface_id: dst_surface_id,
                position: (5.0, 6.0).into(),
                kind: PointerEventKind::Press {
                    button: 0x110,
                    serial: 301,
                },
            },
            PointerEvent {
                surface_id: dst_surface_id,
                position: (5.0, 6.0).into(),
                kind: PointerEventKind::Release {
                    button: 0x110,
                    serial: 302,
                },
            },
        ]));
        h.roundtrip(&mut dst_client);
        assert!(
            dst_client
                .data
                .events
                .iter()
                .any(|e| matches!(e, RecordedEvent::PointerEnter(_)))
        );
        assert!(dst_client.data.events.iter().any(|e| matches!(
            e,
            RecordedEvent::PointerButton {
                button: 0x110,
                pressed: true,
                ..
            }
        )));
        assert!(dst_client.data.events.iter().any(|e| matches!(
            e,
            RecordedEvent::PointerButton {
                button: 0x110,
                pressed: false,
                ..
            }
        )));
        dst_client.data.events.clear();

        // 7. Concurrent multi-MIME receive on destination (replied out-of-order).
        let (r1, w1) = unistd::pipe2(OFlag::O_CLOEXEC).unwrap();
        let (r2, w2) = unistd::pipe2(OFlag::O_CLOEXEC).unwrap();
        offer.receive("text/plain".to_string(), w1.as_fd());
        offer.receive("text/uri-list".to_string(), w2.as_fd());
        drop((w1, w2));
        h.roundtrip(&mut dst_client);

        let xfer_reqs = h.drain_requests();
        assert_eq!(xfer_reqs.len(), 2);
        let (tid1, tid2) = match (&xfer_reqs[0], &xfer_reqs[1]) {
            (
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::RequestDataTransfer {
                        target: DataTransferTarget::DnD(50),
                        mime_type: m1,
                        transfer_id: t1,
                    },
                )),
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::RequestDataTransfer {
                        target: DataTransferTarget::DnD(50),
                        mime_type: m2,
                        transfer_id: t2,
                    },
                )),
            ) => {
                assert_eq!(m1, "text/plain");
                assert_eq!(m2, "text/uri-list");
                assert_ne!(t1, t2);
                (*t1, *t2)
            },
            other => panic!("unexpected transfer requests: {other:?}"),
        };
        // Deliver out of order: tid2 first, then tid1.
        h.send_event(Event::Data(DataEvent::TransferData(
            tid2,
            DataToTransfer(b"file:///tmp/a.txt\r\n".to_vec()),
        )));
        h.send_event(Event::Data(DataEvent::TransferData(
            tid1,
            DataToTransfer(b"hello dnd".to_vec()),
        )));
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        File::from(r1).read_to_end(&mut buf1).unwrap();
        File::from(r2).read_to_end(&mut buf2).unwrap();
        assert_eq!(buf1, b"hello dnd");
        assert_eq!(buf2, b"file:///tmp/a.txt\r\n");

        // 8. Concurrent multi-MIME send from source client.
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::MimeTypeSendRequestedByDestination {
                target: DataTransferTarget::DnD(source_id),
                mime_type: "text/plain".to_string(),
                transfer_id: 701,
            },
        )));
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::MimeTypeSendRequestedByDestination {
                target: DataTransferTarget::DnD(source_id),
                mime_type: "text/uri-list".to_string(),
                transfer_id: 702,
            },
        )));
        h.roundtrip(&mut src_client);
        assert_eq!(src_client.data.send_fds.len(), 2);
        for (mime, fd) in src_client.data.send_fds.drain(..) {
            let payload = if mime == "text/plain" {
                &b"src-plain"[..]
            } else {
                &b"file:///src"[..]
            };
            File::from(fd).write_all(payload).unwrap();
        }
        let mut got = HashMap::new();
        for _ in 0..2 {
            match h.recv_request_timeout(Duration::from_secs(2)) {
                Request::Data(DataRequest::TransferData(tid, data)) => {
                    got.insert(tid, data.0);
                },
                other => panic!("expected TransferData, got {other:?}"),
            }
        }
        assert_eq!(got.get(&701).unwrap(), b"src-plain");
        assert_eq!(got.get(&702).unwrap(), b"file:///src");

        // 9. Destination finishes -> host replies DnDFinished -> source gets
        // ONLY dnd_finished (no spurious cancelled!).
        offer.finish();
        offer.destroy();
        h.roundtrip(&mut dst_client);
        assert_eq!(
            h.drain_requests(),
            vec![Request::Data(DataRequest::DestinationRequest(
                DataDestinationRequest::DnDFinish { offer_id: 50 },
            ))]
        );

        src_client.data.events.clear();
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::DnDFinished { source_id },
        )));
        h.roundtrip(&mut src_client);
        assert_eq!(src_client.data.events, vec![RecordedEvent::SourceFinished]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dnd_ask_action_post_drop_accept_cancel_and_invalid_finish() {
        let mut h = DndHarness::new(true);
        let mut dst_client = h.add_client();
        let dst_surface_id = dst_client.surface_id;

        // Subcase A: Drop with Ask -> synchronous post-drop set_actions(Move) +
        // accept + finish() in a single flush (Dolphin popup menu).
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 10,
                serial: 100,
                surface: dst_surface_id,
                loc: (10.0, 10.0).into(),
                source_actions: (DndAction::Copy | DndAction::Move | DndAction::Ask).into(),
                selected_action: DndAction::Ask.into(),
                mime_types: vec!["text/uri-list".to_string()],
            }),
        )));
        h.roundtrip(&mut dst_client);
        let offer1 = dst_client.data.last_offer.clone().unwrap();
        let enter_serial = dst_client
            .data
            .events
            .iter()
            .find_map(|e| match *e {
                RecordedEvent::DeviceEnter { serial, .. } => Some(serial),
                _ => None,
            })
            .unwrap();
        offer1.accept(enter_serial, Some("text/uri-list".to_string()));
        offer1.set_actions(
            ClientDndAction::Copy | ClientDndAction::Move | ClientDndAction::Ask,
            ClientDndAction::Ask,
        );
        h.roundtrip(&mut dst_client);
        h.drain_requests();

        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDActionSelected {
                offer_id: 10,
                action: DndAction::Ask.into(),
            },
        )));
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDDrop { offer_id: 10 },
        )));
        h.roundtrip(&mut dst_client);
        assert!(!h.state.data_device_state.is_proxy_dnd_grab_active());

        // Single flush post-drop resolution: set_actions(Move, Move) + finish()
        // before host DnDActionSelected round-trip completes.
        offer1.set_actions(ClientDndAction::Move, ClientDndAction::Move);
        offer1.accept(enter_serial, Some("text/uri-list".to_string()));
        offer1.finish();
        offer1.destroy();
        h.roundtrip(&mut dst_client);
        assert_eq!(
            h.drain_requests(),
            vec![
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::DnDSetDestinationActions {
                        offer_id: 10,
                        dnd_actions: DndAction::Move.into(),
                        preferred_action: DndAction::Move.into(),
                    },
                )),
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::DnDAcceptMimeType {
                        offer_id: 10,
                        mime_type: Some("text/uri-list".to_string()),
                    },
                )),
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::DnDFinish { offer_id: 10 },
                )),
            ]
        );

        // Subcase B: Drop with Ask -> user cancels popup -> offer.destroy()
        // without finish() emits DnDCancel so the source doesn't hang.
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 11,
                serial: 101,
                surface: dst_surface_id,
                loc: (10.0, 10.0).into(),
                source_actions: (DndAction::Copy | DndAction::Move | DndAction::Ask).into(),
                selected_action: DndAction::Ask.into(),
                mime_types: vec!["text/uri-list".to_string()],
            }),
        )));
        h.roundtrip(&mut dst_client);
        let offer2 = dst_client.data.last_offer.clone().unwrap();
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDDrop { offer_id: 11 },
        )));
        h.roundtrip(&mut dst_client);
        h.drain_requests();

        offer2.destroy();
        h.roundtrip(&mut dst_client);
        assert_eq!(
            h.drain_requests(),
            vec![Request::Data(DataRequest::DestinationRequest(
                DataDestinationRequest::DnDCancel { offer_id: 11 },
            ))]
        );

        // Subcase C: Post-drop set_actions(None, None) + finish() must be
        // rejected with `wl_data_offer::Error::InvalidFinish`, not allowed via
        // stale `host_selected_action`.
        let mut bad_client = h.add_client();
        let bad_surface_id = bad_client.surface_id;
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 12,
                serial: 102,
                surface: bad_surface_id,
                loc: (10.0, 10.0).into(),
                source_actions: (DndAction::Copy | DndAction::Move).into(),
                selected_action: DndAction::Copy.into(),
                mime_types: vec!["text/plain".to_string()],
            }),
        )));
        h.roundtrip(&mut bad_client);
        let offer3 = bad_client.data.last_offer.clone().unwrap();
        offer3.accept(1, Some("text/plain".to_string()));
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDDrop { offer_id: 12 },
        )));
        h.roundtrip(&mut bad_client);
        h.drain_requests();

        offer3.set_actions(ClientDndAction::None, ClientDndAction::None);
        offer3.finish();
        bad_client.conn.flush().unwrap();
        h.display.dispatch_clients(&mut h.state).unwrap();
        h.display.flush_clients().unwrap();
        let err = bad_client
            .event_queue
            .blocking_dispatch(&mut bad_client.data)
            .unwrap_err();
        let err_str = format!("{err:?}");
        assert!(
            err_str.contains("InvalidFinish")
                || (err_str.contains("wl_data_offer") && err_str.contains("code: 0")),
            "expected InvalidFinish protocol error, got {err_str}"
        );
        assert!(
            !h.drain_requests().iter().any(|r| matches!(
                r,
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::DnDFinish { .. }
                ))
            )),
            "rejected finish must not emit DnDFinish"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_host_to_remote_hover_and_leave_releases_grab() {
        let mut h = DndHarness::new(true);
        let mut dst_client = h.add_client();
        let dst_surface_id = dst_client.surface_id;

        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 1,
                serial: 50,
                surface: dst_surface_id,
                loc: (12.0, 34.0).into(),
                source_actions: DndAction::Copy.into(),
                selected_action: DndAction::Copy.into(),
                mime_types: vec!["text/plain".to_string()],
            }),
        )));
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDMotion {
                offer_id: 1,
                loc: (20.0, 40.0).into(),
            },
        )));
        h.roundtrip(&mut dst_client);
        assert!(h.state.data_device_state.is_proxy_dnd_grab_active());
        dst_client.data.events.clear();

        // Drag leaves without dropping.
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDLeave { offer_id: 1 },
        )));
        h.roundtrip(&mut dst_client);
        assert_eq!(dst_client.data.events, vec![RecordedEvent::DeviceLeave]);
        assert!(!h.state.data_device_state.is_proxy_dnd_grab_active());
        dst_client.data.events.clear();

        // Subsequent normal pointer enter/motion/button works immediately.
        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id: dst_surface_id,
                position: (15.0, 25.0).into(),
                kind: PointerEventKind::Enter { serial: 51 },
            },
            PointerEvent {
                surface_id: dst_surface_id,
                position: (16.0, 26.0).into(),
                kind: PointerEventKind::Motion,
            },
        ]));
        h.roundtrip(&mut dst_client);
        assert!(
            dst_client
                .data
                .events
                .iter()
                .any(|e| matches!(e, RecordedEvent::PointerEnter(_)))
        );
        assert!(
            dst_client
                .data
                .events
                .contains(&RecordedEvent::PointerMotion(16.0, 26.0))
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::verbose_file_reads)]
    fn test_start_drag_rejection_source_destroy_and_reconnect_cleanup() {
        let mut h = DndHarness::new(true);
        let mut client = h.add_client();
        let surface_id = client.surface_id;
        h.drain_requests();

        // 1. start_drag without an implicit grab must immediately emit
        // `wl_data_source.cancelled` (fixes Smithay#1887 / Electron hang).
        let bad_source = client
            .data
            .data_device_manager
            .as_ref()
            .unwrap()
            .create_data_source(&client.qh, ());
        client
            .data_device
            .start_drag(Some(&bad_source), &client.surface, None, 9999);
        h.roundtrip(&mut client);
        assert_eq!(client.data.events, vec![RecordedEvent::SourceCancelled]);
        assert!(h.drain_requests().is_empty());
        client.data.events.clear();

        // 2. Client destroying active WlDataSource mid-drag sends CancelDrag
        // and unsets ProxyDndGrab.
        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id,
                position: (5.0, 5.0).into(),
                kind: PointerEventKind::Enter { serial: 10 },
            },
            PointerEvent {
                surface_id,
                position: (5.0, 5.0).into(),
                kind: PointerEventKind::Press {
                    button: 0x110,
                    serial: 11,
                },
            },
        ]));
        h.roundtrip(&mut client);
        let press_serial = client
            .data
            .events
            .iter()
            .find_map(|e| match *e {
                RecordedEvent::PointerButton {
                    serial,
                    pressed: true,
                    ..
                } => Some(serial),
                _ => None,
            })
            .unwrap();
        client.data.events.clear();

        let source = client
            .data
            .data_device_manager
            .as_ref()
            .unwrap()
            .create_data_source(&client.qh, ());
        source.offer("text/plain".to_string());
        source.set_actions(ClientDndAction::Copy);
        client
            .data_device
            .start_drag(Some(&source), &client.surface, None, press_serial);
        h.roundtrip(&mut client);
        let source_id = h
            .drain_requests()
            .into_iter()
            .find_map(|r| match r {
                Request::Data(DataRequest::SourceRequest(DataSourceRequest::StartDrag(sd))) => {
                    Some(sd.source_id)
                },
                _ => None,
            })
            .unwrap();
        assert!(h.state.data_device_state.is_proxy_dnd_grab_active());

        source.destroy();
        h.roundtrip(&mut client);
        assert!(!h.state.data_device_state.is_proxy_dnd_grab_active());
        assert_eq!(
            h.drain_requests(),
            vec![Request::Data(DataRequest::SourceRequest(
                DataSourceRequest::CancelDrag(source_id),
            ))]
        );

        // 3. Stale send request for cancelled source immediately replies with
        // empty TransferData so wprsc closes the host write pipe.
        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::MimeTypeSendRequestedByDestination {
                target: DataTransferTarget::DnD(source_id),
                mime_type: "text/plain".to_string(),
                transfer_id: 88,
            },
        )));
        assert_eq!(
            h.drain_requests(),
            vec![Request::Data(DataRequest::TransferData(
                88,
                DataToTransfer(Vec::new()),
            ))]
        );

        // 4. Reconnect mid-drag cancels source, leaves incoming DnD, unsets
        // grab, and closes pending receive pipes with EOF.
        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id,
                position: (5.0, 5.0).into(),
                kind: PointerEventKind::Enter { serial: 20 },
            },
            PointerEvent {
                surface_id,
                position: (5.0, 5.0).into(),
                kind: PointerEventKind::Press {
                    button: 0x110,
                    serial: 21,
                },
            },
        ]));
        h.roundtrip(&mut client);
        let press_serial2 = client
            .data
            .events
            .iter()
            .find_map(|e| match *e {
                RecordedEvent::PointerButton {
                    serial,
                    pressed: true,
                    ..
                } => Some(serial),
                _ => None,
            })
            .unwrap();
        client.data.events.clear();

        let source2 = client
            .data
            .data_device_manager
            .as_ref()
            .unwrap()
            .create_data_source(&client.qh, ());
        source2.offer("text/plain".to_string());
        source2.set_actions(ClientDndAction::Copy);
        client
            .data_device
            .start_drag(Some(&source2), &client.surface, None, press_serial2);
        h.roundtrip(&mut client);
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::DnDEnter(DragEnter {
                offer_id: 77,
                serial: 22,
                surface: surface_id,
                loc: (5.0, 5.0).into(),
                source_actions: DndAction::Copy.into(),
                selected_action: DndAction::Copy.into(),
                mime_types: vec!["text/plain".to_string()],
            }),
        )));
        h.roundtrip(&mut client);
        let offer = client.data.last_offer.clone().unwrap();
        let (read_fd, write_fd) = unistd::pipe2(OFlag::O_CLOEXEC).unwrap();
        offer.receive("text/plain".to_string(), write_fd.as_fd());
        drop(write_fd);
        h.roundtrip(&mut client);
        client.data.events.clear();

        h.connect_wprsc();
        h.roundtrip(&mut client);
        assert!(!h.state.data_device_state.is_proxy_dnd_grab_active());
        assert!(client.data.events.contains(&RecordedEvent::SourceCancelled));
        assert!(client.data.events.contains(&RecordedEvent::DeviceLeave));
        let mut buf = Vec::new();
        File::from(read_fd).read_to_end(&mut buf).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_internal_drag_without_source_delivers_drop_and_restores_pointer() {
        let mut h = DndHarness::new(true);
        let mut client = h.add_client();
        let surface_id = client.surface_id;
        h.drain_requests();

        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id,
                position: (10.0, 10.0).into(),
                kind: PointerEventKind::Enter { serial: 1 },
            },
            PointerEvent {
                surface_id,
                position: (10.0, 10.0).into(),
                kind: PointerEventKind::Press {
                    button: 0x110,
                    serial: 2,
                },
            },
        ]));
        h.roundtrip(&mut client);
        let press_serial = client
            .data
            .events
            .iter()
            .find_map(|e| match *e {
                RecordedEvent::PointerButton {
                    serial,
                    pressed: true,
                    ..
                } => Some(serial),
                _ => None,
            })
            .unwrap();
        client.data.events.clear();
        h.drain_requests();

        client
            .data_device
            .start_drag(None, &client.surface, None, press_serial);
        h.roundtrip(&mut client);
        assert!(
            !h.drain_requests()
                .iter()
                .any(|r| matches!(r, Request::Data(_))),
            "internal drag without source must not proxy DataRequest to host"
        );

        h.send_event(Event::PointerFrame(vec![
            PointerEvent {
                surface_id,
                position: (20.0, 30.0).into(),
                kind: PointerEventKind::Motion,
            },
            PointerEvent {
                surface_id,
                position: (25.0, 35.0).into(),
                kind: PointerEventKind::Motion,
            },
            PointerEvent {
                surface_id,
                position: (25.0, 35.0).into(),
                kind: PointerEventKind::Release {
                    button: 0x110,
                    serial: 3,
                },
            },
        ]));
        h.roundtrip(&mut client);

        assert!(client.data.events.iter().any(|e| matches!(
            e,
            RecordedEvent::DeviceEnter {
                has_offer: false,
                x,
                y,
                ..
            } if (*x, *y) == (20.0, 30.0)
        )));
        assert!(
            client
                .data
                .events
                .contains(&RecordedEvent::DeviceMotion(25.0, 35.0))
        );
        assert!(client.data.events.contains(&RecordedEvent::DeviceDrop));
        assert!(client.data.events.contains(&RecordedEvent::DeviceLeave));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::verbose_file_reads)]
    fn test_clipboard_selection_and_concurrent_receive() {
        let mut h = DndHarness::new(true);
        let mut client = h.add_client();
        let surface_id = client.surface_id;
        h.drain_requests();

        // Give client keyboard focus so it receives clipboard offers.
        h.send_event(Event::KeyboardEvent(KeyboardEvent::Enter {
            serial: 1,
            surface_id,
            keycodes: Vec::new(),
            keysyms: Vec::new(),
        }));
        h.send_event(Event::Data(DataEvent::DestinationEvent(
            DataDestinationEvent::SelectionSet(
                DataSource::Selection,
                SourceMetadata::from_mime_types(vec![
                    "text/plain".to_string(),
                    "text/html".to_string(),
                ]),
            ),
        )));
        h.roundtrip(&mut client);
        assert!(
            client
                .data
                .events
                .contains(&RecordedEvent::DeviceSelection(true))
        );
        let offer = client.data.last_offer.clone().unwrap();

        let (r1, w1) = unistd::pipe2(OFlag::O_CLOEXEC).unwrap();
        let (r2, w2) = unistd::pipe2(OFlag::O_CLOEXEC).unwrap();
        offer.receive("text/plain".to_string(), w1.as_fd());
        offer.receive("text/html".to_string(), w2.as_fd());
        drop((w1, w2));
        h.roundtrip(&mut client);

        let reqs = h.drain_requests();
        assert_eq!(reqs.len(), 2);
        let (tid1, tid2) = match (&reqs[0], &reqs[1]) {
            (
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::RequestDataTransfer {
                        target: DataTransferTarget::Selection,
                        transfer_id: t1,
                        ..
                    },
                )),
                Request::Data(DataRequest::DestinationRequest(
                    DataDestinationRequest::RequestDataTransfer {
                        target: DataTransferTarget::Selection,
                        transfer_id: t2,
                        ..
                    },
                )),
            ) => (*t1, *t2),
            other => panic!("unexpected requests: {other:?}"),
        };
        h.send_event(Event::Data(DataEvent::TransferData(
            tid1,
            DataToTransfer(b"plain-clip".to_vec()),
        )));
        h.send_event(Event::Data(DataEvent::TransferData(
            tid2,
            DataToTransfer(b"<b>html</b>".to_vec()),
        )));
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        File::from(r1).read_to_end(&mut buf1).unwrap();
        File::from(r2).read_to_end(&mut buf2).unwrap();
        assert_eq!(buf1, b"plain-clip");
        assert_eq!(buf2, b"<b>html</b>");

        // Client sets selection -> forwarded to host, and host can read from it.
        let source = client
            .data
            .data_device_manager
            .as_ref()
            .unwrap()
            .create_data_source(&client.qh, ());
        source.offer("text/plain".to_string());
        client.data_device.set_selection(Some(&source), 1);
        h.roundtrip(&mut client);
        assert_eq!(
            h.drain_requests(),
            vec![Request::Data(DataRequest::SourceRequest(
                DataSourceRequest::SetSelection(
                    DataSource::Selection,
                    SourceMetadata::from_mime_types(vec!["text/plain".to_string()]),
                ),
            ))]
        );

        h.send_event(Event::Data(DataEvent::SourceEvent(
            DataSourceEvent::MimeTypeSendRequestedByDestination {
                target: DataTransferTarget::Selection,
                mime_type: "text/plain".to_string(),
                transfer_id: 900,
            },
        )));
        h.roundtrip(&mut client);
        assert_eq!(client.data.send_fds.len(), 1);
        let (mime, fd) = client.data.send_fds.pop().unwrap();
        assert_eq!(mime, "text/plain");
        File::from(fd).write_all(b"from-client").unwrap();
        assert_eq!(
            h.recv_request_timeout(Duration::from_secs(2)),
            Request::Data(DataRequest::TransferData(
                900,
                DataToTransfer(b"from-client".to_vec()),
            ))
        );
    }
}
