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

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;
use std::net::Shutdown;
use std::os::fd::AsFd;
use std::os::unix::net::UnixListener;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process;
use std::str;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::thread::Scope;
use std::thread::ScopedJoinHandle;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use nix::sys::socket;
use nix::sys::socket::sockopt::RcvBuf;
use nix::sys::socket::sockopt::SndBuf;
use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;
use rkyv::Archive;
use rkyv::Deserialize;
use rkyv::Serialize;
use rkyv::api::high::HighDeserializer;
use rkyv::api::high::HighSerializer;
use rkyv::api::high::HighValidator;
use rkyv::bytecheck;
use rkyv::rancor::Error as RancorError;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use smithay::reexports::calloop::channel;
use smithay::reexports::calloop::channel::Channel;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::backend;
use sysctl::Ctl;
use sysctl::Sysctl;

use crate::prelude::*;
use crate::utils;

pub mod framing;
pub mod geometry;
pub mod tuple;
pub mod wayland;
pub mod xdg_shell;

use framing::CompressedTileStream;
use framing::Framed;
use framing::ObjectCompressor;
use framing::ObjectDecompressor;
use framing::StreamingCompressedTiles;

#[derive(Archive, Deserialize, Serialize, Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct ClientId(pub u64);

impl ClientId {
    pub fn new(client: &Client) -> Self {
        Self(hash(&client.id()))
    }
}

impl From<backend::ClientId> for ClientId {
    fn from(client_id: backend::ClientId) -> Self {
        (&client_id).into()
    }
}

impl From<&backend::ClientId> for ClientId {
    fn from(client_id: &backend::ClientId) -> Self {
        Self(hash(client_id))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Archive, Deserialize, Serialize, serde_derive::Serialize)]
pub struct Capabilities {
    pub xwayland: bool,
}

// TODO: https://github.com/rust-lang/rfcs/pull/2593 - simplify all the enums.

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub enum Request {
    Surface(wayland::SurfaceRequest),
    CursorImage(wayland::CursorImage),
    Toplevel(xdg_shell::ToplevelRequest),
    Popup(xdg_shell::PopupRequest),
    Data(wayland::DataRequest),
    ClientDisconnected(ClientId),
    Capabilities(Capabilities),
}

#[derive(Debug, Clone, PartialEq, Archive, Deserialize, Serialize)]
pub enum Event {
    Output(wayland::OutputEvent),
    PointerFrame(Vec<wayland::PointerEvent>),
    KeyboardEvent(wayland::KeyboardEvent),
    Toplevel(xdg_shell::ToplevelEvent),
    Popup(xdg_shell::PopupEvent),
    Data(wayland::DataEvent),
    Surface(wayland::SurfaceEvent),
}

// TODO: test that object ids with same value from different clients hash
// differently.
pub fn hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

const CHANNEL_SIZE: usize = 1024;

pub trait Serializable:
    Debug
    + Send
    + Archive
    + for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, RancorError>>
    + 'static
{
}

impl<T> Serializable for T where
    T: Debug
        + Send
        + Archive
        + for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, RancorError>>
        + 'static
{
}

fn socket_buffer_limits() -> Result<(usize, usize)> {
    let rmem_max: usize = Ctl::new("net.core.rmem_max")
        .location(loc!())?
        .value_string()
        .location(loc!())?
        .parse()
        .location(loc!())?;
    let wmem_max: usize = Ctl::new("net.core.wmem_max")
        .location(loc!())?
        .value_string()
        .location(loc!())?
        .parse()
        .location(loc!())?;
    Ok((rmem_max, wmem_max))
}

fn enlarge_socket_buffer<F: AsFd>(fd: &F) {
    let (rmem_max, wmem_max) = warn_and_return!(socket_buffer_limits());

    socket::setsockopt(fd, RcvBuf, &rmem_max).warn_and_ignore(loc!());
    socket::setsockopt(fd, SndBuf, &wmem_max).warn_and_ignore(loc!());
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Version(String);

impl Version {
    fn new() -> Self {
        Self(env!("SERIALIZATION_TREE_HASH").to_string())
    }

    fn compare_and_warn(&self, other: &Self) {
        if self != other {
            warn!(
                "Self version is {:?}, while other version is {:?}. These versions may be incompatible; if you experience bugs (especially hanging or crashes), restart the server.",
                self, other
            );
        }
    }
}

impl Framed for Version {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        self.0.framed_write(stream)
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        Ok(Self(String::framed_read(stream).location(loc!())?))
    }
}

// TODO: figure out how to shorten the T::Archived bound. This may require
// https://github.com/rust-lang/rust/issues/52662.

pub enum SendType<ST>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    Object(ST),
    RawBuffer(StreamingCompressedTiles),
}

impl<ST> fmt::Debug for SendType<ST>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Object(obj) => write!(f, "Object({obj:?})"),
            Self::RawBuffer(tiles) => {
                write!(f, "RawBuffer([{:?}])", tiles.uncompressed_size())
            },
        }
    }
}

pub enum RecvType<RT>
where
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    Connected,
    Object(RT),
    RawBuffer(CompressedTileStream),
}

impl<RT> fmt::Debug for RecvType<RT>
where
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connected => write!(f, "Connected"),
            Self::Object(obj) => write!(f, "Object({obj:?})"),
            Self::RawBuffer(stream) => write!(f, "RawBuffer([{:?}])", stream.header),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum MessageType {
    Object,
    RawBuffer,
}

impl Framed for MessageType {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        let val: u8 = (*self).into();
        val.framed_write(stream)
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        Self::try_from(u8::framed_read(stream).location(loc!())?).location(loc!())
    }
}

enum ConnectionSession<T> {
    Disconnected,
    Pending { conn_id: u64, sender: Sender<T> },
    Active { conn_id: u64, sender: Sender<T> },
}

struct ConnectionState<T> {
    next_conn_id: u64,
    session: ConnectionSession<T>,
}

struct ConnectionSlot<T> {
    state: Mutex<ConnectionState<T>>,
}

impl<T> ConnectionSlot<T> {
    fn new_server() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(ConnectionState {
                next_conn_id: 0,
                session: ConnectionSession::Disconnected,
            }),
        })
    }

    fn new_client(sender: Sender<T>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(ConnectionState {
                next_conn_id: 1,
                session: ConnectionSession::Active { conn_id: 1, sender },
            }),
        })
    }

    fn begin_connection(&self, sender: Sender<T>) -> u64 {
        let mut guard = self.state.lock().unwrap();
        guard.next_conn_id += 1;
        let conn_id = guard.next_conn_id;
        guard.session = ConnectionSession::Pending { conn_id, sender };
        conn_id
    }

    fn activate_connection(&self, conn_id: u64) -> bool {
        let mut guard = self.state.lock().unwrap();
        match &guard.session {
            ConnectionSession::Pending {
                conn_id: pending_id,
                sender,
            } if *pending_id == conn_id => {
                let sender = sender.clone();
                guard.session = ConnectionSession::Active { conn_id, sender };
                true
            },
            _ => false,
        }
    }

    fn is_connected(&self) -> bool {
        matches!(
            self.state.lock().unwrap().session,
            ConnectionSession::Active { .. }
        )
    }

    fn is_active_connection(&self, conn_id: u64) -> bool {
        matches!(
            &self.state.lock().unwrap().session,
            ConnectionSession::Active { conn_id: active_id, .. } if *active_id == conn_id
        )
    }

    fn end_connection(&self, conn_id: u64) {
        let mut guard = self.state.lock().unwrap();
        let matches_id = match &guard.session {
            ConnectionSession::Disconnected => false,
            ConnectionSession::Pending { conn_id: id, .. }
            | ConnectionSession::Active { conn_id: id, .. } => *id == conn_id,
        };
        if matches_id {
            guard.session = ConnectionSession::Disconnected;
        }
    }
}

pub struct SerializerWriter<ST>(Arc<ConnectionSlot<SendType<ST>>>)
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>;

impl<ST> SerializerWriter<ST>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    pub fn send(&self, msg: SendType<ST>) {
        let guard = self.0.state.lock().unwrap();
        if let ConnectionSession::Active { sender, .. } = &guard.session {
            _ = sender.send(msg);
        }
    }
}

impl<ST> Clone for SerializerWriter<ST>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

fn read_loop<R, RT>(
    conn_id: u64,
    stream: R,
    output_channel: channel::SyncSender<(u64, RecvType<RT>)>,
) -> Result<()>
where
    R: Read,
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    let mut stream = BufReader::with_capacity(256 * 1024, stream);
    let mut decompressor = ObjectDecompressor::new().location(loc!())?;

    Version::new().compare_and_warn(&Version::framed_read(&mut stream).location(loc!())?);

    loop {
        let message_type = MessageType::framed_read(&mut stream).location(loc!())?;
        debug!("read message_type: {:?}", message_type);

        // read_exact blocks waiting for data, so start the span afterward.
        let _span = debug_span!("serializer_read_loop").entered();

        match message_type {
            MessageType::Object => {
                decompressor
                    .read_framed_with(&mut stream, |buf| {
                        let obj = RecvType::Object(
                            debug_span!("deserialize")
                                .in_scope(|| rkyv::from_bytes(buf))
                                .location(loc!())?,
                        );
                        debug!("read obj: {obj:?}");
                        output_channel
                            .send((conn_id, obj))
                            // The error type is not Send + Sync, which anyhow requires.
                            .map_err(|e| anyhow!("{e}"))
                            .location(loc!())?;
                        Ok(())
                    })
                    .location(loc!())?;
            },
            MessageType::RawBuffer => {
                CompressedTileStream::read_from_stream(&mut stream, |tile_stream| {
                    let obj = RecvType::RawBuffer(tile_stream);
                    debug!("read obj: {obj:?}");
                    output_channel
                        .send((conn_id, obj))
                        // The error type is not Send + Sync, which anyhow requires.
                        .map_err(|e| anyhow!("{e}"))
                        .location(loc!())
                })
                .location(loc!())?;
            },
        }
    }
}

fn write_loop<W, ST>(stream: W, input_channel: Receiver<SendType<ST>>) -> Result<()>
where
    W: Write,
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    let (_, wmem_max) = socket_buffer_limits().location(loc!())?;
    let mut stream = BufWriter::with_capacity(
        wmem_max, // match the socket's buffer size
        stream,
    );

    let mut compressor = ObjectCompressor::new(1).location(loc!())?;

    Version::new().framed_write(&mut stream).location(loc!())?;
    stream.flush().location(loc!())?;

    while let Ok(obj) = input_channel.recv() {
        debug!("sending obj: {:?}", obj);

        // recv blocks while waiting for data, so start the span afterward.
        let span = debug_span!(
            "serializer_write_loop",
            uncompressed_size = field::Empty,
            compressed_size = field::Empty,
            compression_ratio = field::Empty
        )
        .entered();
        let (uncompressed_size, compressed_size) = match obj {
            SendType::Object(obj) => {
                let serialized_data = debug_span!("serialize")
                    .in_scope(|| rkyv::to_bytes::<RancorError>(&obj))
                    .location(loc!())?;

                MessageType::Object
                    .framed_write(&mut stream)
                    .location(loc!())?;
                let compressed_size = compressor
                    .write_framed(&serialized_data, &mut stream)
                    .location(loc!())?;
                (serialized_data.len(), compressed_size)
            },
            SendType::RawBuffer(streaming_tiles) => {
                let uncompressed_size = streaming_tiles.uncompressed_size();
                MessageType::RawBuffer
                    .framed_write(&mut stream)
                    .location(loc!())?;
                let compressed_size = streaming_tiles.framed_write(&mut stream).location(loc!())?;
                (uncompressed_size, compressed_size)
            },
        };
        stream.flush().location(loc!())?;

        // metrics
        {
            let compression_ratio = if compressed_size > 0 {
                uncompressed_size as f64 / compressed_size as f64
            } else {
                1.0
            };
            span.record("uncompressed_size", field::debug(uncompressed_size));
            span.record("compressed_size", compressed_size);
            span.record("compression_ratio", compression_ratio);

            #[cfg(feature = "tracy")]
            if let Some(tracy_client) = tracy_client::Client::running() {
                tracy_client.plot(
                    tracy_client::plot_name!("compressed_size"),
                    compressed_size as f64,
                );
                tracy_client.plot(
                    tracy_client::plot_name!("compression_ratio"),
                    compression_ratio,
                );
                if compression_ratio > 1.0 {
                    tracy_client.plot(
                        tracy_client::plot_name!("filtered_compression_ratio"),
                        compression_ratio,
                    );
                }
            }
        }
    }
    Ok(())
}

fn spawn_rw_loops<'scope, ST, RT>(
    scope: &'scope Scope<'scope, '_>,
    conn_id: u64,
    stream: UnixStream,
    read_channel_tx: channel::SyncSender<(u64, RecvType<RT>)>,
    write_channel_rx: Receiver<SendType<ST>>,
) -> Result<(
    ScopedJoinHandle<'scope, Result<()>>,
    ScopedJoinHandle<'scope, Result<()>>,
)>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    let read_stream = stream.try_clone().location(loc!())?;
    let read_thread = scope.spawn(move || read_loop(conn_id, read_stream, read_channel_tx));

    let write_stream = stream.try_clone().location(loc!())?;
    let write_thread = scope.spawn(move || write_loop(write_stream, write_channel_rx));

    Ok((read_thread, write_thread))
}

fn accept_loop<ST, RT>(
    listener: UnixListener,
    read_channel_tx: channel::SyncSender<(u64, RecvType<RT>)>,
    slot: Arc<ConnectionSlot<SendType<ST>>>,
) where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    thread::scope(|scope| {
        loop {
            debug!("waiting for client connection");
            let (stream, _) = listener.accept().unwrap();
            info!("wprs client connected");
            let (conn_tx, conn_rx) = crossbeam_channel::unbounded();
            let conn_id = slot.begin_connection(conn_tx);
            // Notify the server event loop immediately on `accept()` rather than
            // waiting for a client-initiated packet: over `ssh -L`, OpenSSH holds
            // client writes until remote `sshd` connects and confirms the channel,
            // so activating on `accept()` saves a full WAN round-trip on attach.
            if read_channel_tx
                .send((conn_id, RecvType::Connected))
                .is_err()
            {
                slot.end_connection(conn_id);
                break;
            }
            let (read_thread, write_thread) = spawn_rw_loops(
                scope,
                conn_id,
                stream.try_clone().unwrap(),
                read_channel_tx.clone(),
                conn_rx,
            )
            .unwrap();
            let read_thread_result = utils::join_unwrap(read_thread);
            debug!("read thread joined: {read_thread_result:?}");
            slot.end_connection(conn_id);
            // Shut down the socket before joining `write_thread` so a `write_all`
            // or `flush` blocked on a full send buffer wakes immediately with
            // `EPIPE`/`ESHUTDOWN` if the peer stopped reading. Ignore `ENOTCONN`
            // if the peer already closed.
            _ = stream.shutdown(Shutdown::Both);
            let write_thread_result = utils::join_unwrap(write_thread);
            debug!("write thread joined: {write_thread_result:?}");
        }
    });
}

fn client_loop<ST, RT>(
    stream: UnixStream,
    read_channel_tx: channel::SyncSender<(u64, RecvType<RT>)>,
    write_channel_rx: Receiver<SendType<ST>>,
) -> Result<()>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    thread::scope(|scope| {
        let (read_thread, _) =
            spawn_rw_loops(scope, 1, stream, read_channel_tx, write_channel_rx).location(loc!())?;

        // TODO: consider actually look at the error and not printing the reason
        // if was actually just a disconnection and not some other error.
        let result = utils::join_unwrap(read_thread);
        debug!("read thread joined: {:?}", result);
        eprintln!("server disconnected: {result:?}");
        process::exit(1);
    })
}

// TODO: can we create a separate thread to handle serialization/deserialization
// for each client? In principle, each client's stream is independent, but what
// about things like setting the cursor? Rather, which client do we associate
// that with? Any client?
pub struct Serializer<ST, RT>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    read_handle: Option<Channel<(u64, RecvType<RT>)>>,
    slot: Arc<ConnectionSlot<SendType<ST>>>,
}

impl<ST, RT> Serializer<ST, RT>
where
    ST: Serializable,
    ST::Archived: Deserialize<ST, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
    RT: Serializable,
    RT::Archived: Deserialize<RT, HighDeserializer<RancorError>>
        + for<'a> bytecheck::CheckBytes<HighValidator<'a, RancorError>>,
{
    pub fn new_server<P: AsRef<Path>>(sock_path: P) -> Result<Self> {
        let listener = utils::bind_user_socket(sock_path).location(loc!())?;
        enlarge_socket_buffer(&listener);

        let (reader_tx, reader_rx): (
            channel::SyncSender<(u64, RecvType<RT>)>,
            Channel<(u64, RecvType<RT>)>,
        ) = channel::sync_channel(CHANNEL_SIZE);
        let slot = ConnectionSlot::new_server();

        {
            let slot = slot.clone();
            thread::spawn(move || accept_loop(listener, reader_tx, slot));
        }

        Ok(Self {
            read_handle: Some(reader_rx),
            slot,
        })
    }

    pub fn new_client<P: AsRef<Path>>(sock_path: P) -> Result<Self> {
        let stream = UnixStream::connect(sock_path).location(loc!())?;
        enlarge_socket_buffer(&stream);

        let (reader_tx, reader_rx): (
            channel::SyncSender<(u64, RecvType<RT>)>,
            Channel<(u64, RecvType<RT>)>,
        ) = channel::sync_channel(CHANNEL_SIZE);
        let (writer_tx, writer_rx): (Sender<SendType<ST>>, Receiver<SendType<ST>>) =
            crossbeam_channel::unbounded();
        let slot = ConnectionSlot::new_client(writer_tx);

        thread::spawn(move || client_loop(stream, reader_tx, writer_rx));

        Ok(Self {
            read_handle: Some(reader_rx),
            slot,
        })
    }

    // TODO: https://github.com/rust-lang/rfcs/issues/1215 - Ideally this would
    // return an &mut, but we can't afford to tie up the entire serializer for,
    // well, ever. Change this to return an &mut once rust supports partial
    // borrowing of struct fields.
    // TODO: rename to receiver.
    pub fn reader(&mut self) -> Option<Channel<(u64, RecvType<RT>)>> {
        self.read_handle.take()
    }

    pub fn writer(&self) -> SerializerWriter<ST> {
        SerializerWriter(self.slot.clone())
    }

    pub fn other_end_connected(&self) -> bool {
        self.slot.is_connected()
    }

    pub fn activate_connection(&mut self, conn_id: u64) -> bool {
        self.slot.activate_connection(conn_id)
    }

    pub fn is_active_connection(&self, conn_id: u64) -> bool {
        self.slot.is_active_connection(conn_id)
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::TryRecvError;

    use super::*;

    #[test]
    fn test_connection_slot_session_gating() {
        let slot: Arc<ConnectionSlot<SendType<u32>>> = ConnectionSlot::new_server();
        let writer = SerializerWriter(slot.clone());

        // 1. Detached server discards sends and reports disconnected.
        assert!(!slot.is_connected());
        assert!(!slot.is_active_connection(1));
        writer.send(SendType::Object(10));

        // 2. Pending session keeps `is_connected() == false` and discards pre-activation sends.
        let (tx1, rx1) = crossbeam_channel::unbounded();
        let c1 = slot.begin_connection(tx1);
        assert_eq!(c1, 1);
        assert!(!slot.is_connected());
        assert!(!slot.is_active_connection(c1));
        writer.send(SendType::Object(20));
        assert!(matches!(rx1.try_recv(), Err(TryRecvError::Empty)));

        // 3. Ending a pending connection before activation drops sender and rejects late activation.
        slot.end_connection(c1);
        assert!(matches!(rx1.try_recv(), Err(TryRecvError::Disconnected)));
        assert!(!slot.activate_connection(c1));
        assert!(!slot.is_connected());

        // 4. Activating `c2` delivers messages once and rejects duplicate activation.
        let (tx2, rx2) = crossbeam_channel::unbounded();
        let c2 = slot.begin_connection(tx2);
        assert_eq!(c2, 2);
        assert!(slot.activate_connection(c2));
        assert!(!slot.activate_connection(c2));
        assert!(slot.is_connected());
        assert!(slot.is_active_connection(c2));
        writer.send(SendType::Object(30));
        assert!(matches!(rx2.try_recv(), Ok(SendType::Object(30))));

        // 5. Rapid reconnect `c3` immediately closes `rx2`, ignores stale `c2`
        // calls, and buffers nothing until `c3` activates.
        let (tx3, rx3) = crossbeam_channel::unbounded();
        let c3 = slot.begin_connection(tx3);
        assert_eq!(c3, 3);
        assert!(!slot.is_connected());
        assert!(!slot.is_active_connection(c2));
        assert!(matches!(rx2.try_recv(), Err(TryRecvError::Disconnected)));
        writer.send(SendType::Object(40));
        assert!(matches!(rx3.try_recv(), Err(TryRecvError::Empty)));
        slot.end_connection(c2);
        assert!(!slot.activate_connection(c2));
        assert!(slot.activate_connection(c3));
        assert!(!slot.is_active_connection(c2));
        slot.end_connection(c2);
        assert!(slot.is_connected());
        assert!(slot.is_active_connection(c3));
        writer.send(SendType::Object(50));
        assert!(matches!(rx3.try_recv(), Ok(SendType::Object(50))));

        // 6. Ending `c3` drops `tx3` and leaves `writer.send` a safe no-op.
        slot.end_connection(c3);
        assert!(!slot.is_connected());
        assert!(!slot.is_active_connection(c3));
        assert!(matches!(rx3.try_recv(), Err(TryRecvError::Disconnected)));
        writer.send(SendType::Object(60));

        // 7. Client slot starts active immediately.
        let (tx_client, rx_client) = crossbeam_channel::unbounded();
        let client_slot: Arc<ConnectionSlot<SendType<u32>>> = ConnectionSlot::new_client(tx_client);
        let client_writer = SerializerWriter(client_slot.clone());
        assert!(client_slot.is_connected());
        assert!(client_slot.is_active_connection(1));
        assert!(!client_slot.is_active_connection(2));
        client_writer.send(SendType::Object(70));
        assert!(matches!(rx_client.try_recv(), Ok(SendType::Object(70))));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_server_accept_and_reconnect_teardown() {
        let sock_path = std::env::temp_dir().join(format!(
            "wprs_accept_test_{}_{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut server: Serializer<Request, Event> = Serializer::new_server(&sock_path).unwrap();
        let reader = server.reader().unwrap();

        let mut stream = UnixStream::connect(&sock_path).unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let (conn_id, msg) = reader.recv().unwrap();
        assert_eq!(conn_id, 1);
        assert!(matches!(msg, RecvType::Connected));
        assert!(server.activate_connection(conn_id));
        assert!(server.other_end_connected());

        // Drain the initial handshake version frame, then exercise server
        // `write_loop` framing + per-message `flush()` (without which this
        // small object stays buffered in `BufWriter` and hits the read
        // timeout).
        assert_eq!(Version::framed_read(&mut stream).unwrap(), Version::new());
        let expected_req = Request::ClientDisconnected(ClientId(7));
        server.writer().send(SendType::Object(expected_req.clone()));
        assert_eq!(
            MessageType::framed_read(&mut stream).unwrap(),
            MessageType::Object
        );
        let mut decompressor = ObjectDecompressor::new().unwrap();
        let mut decoded_req = None;
        decompressor
            .read_framed_with(&mut stream, |buf| {
                decoded_req = Some(rkyv::from_bytes::<Request, RancorError>(buf).location(loc!())?);
                Ok(())
            })
            .unwrap();
        assert_eq!(decoded_req, Some(expected_req));

        // Dropping `stream` ends `read_loop`, so `write_loop` (parked in
        // `recv()`) can only exit when `end_connection` drops the sender;
        // `(2, Connected)` arriving below verifies `accept_loop` joined both
        // threads and accepted the reconnect.
        drop(stream);

        let mut stream2 = UnixStream::connect(&sock_path).unwrap();
        let (conn_id2, msg2) = reader.recv().unwrap();
        assert_eq!(conn_id2, 2);
        assert!(matches!(msg2, RecvType::Connected));
        assert!(server.activate_connection(conn_id2));
        assert!(server.other_end_connected());

        // Exercise `read_loop`'s `(conn_id, Object)` path on the reconnect
        // session so passing a stale/wrong `conn_id` to `spawn_rw_loops` fails.
        Version::new().framed_write(&mut stream2).unwrap();
        let expected_event = Event::Surface(wayland::SurfaceEvent {
            surface_id: wayland::WlSurfaceId(11),
            payload: wayland::SurfaceEventPayload::BufferResyncRequested,
        });
        let serialized_event = rkyv::to_bytes::<RancorError>(&expected_event).unwrap();
        MessageType::Object.framed_write(&mut stream2).unwrap();
        let mut compressor = ObjectCompressor::new(1).unwrap();
        compressor
            .write_framed(&serialized_event, &mut stream2)
            .unwrap();
        let (event_conn_id, event_msg) = reader.recv().unwrap();
        assert_eq!(event_conn_id, 2);
        assert!(matches!(event_msg, RecvType::Object(ev) if ev == expected_event));

        drop(stream2);
        _ = std::fs::remove_file(&sock_path);
    }
}
