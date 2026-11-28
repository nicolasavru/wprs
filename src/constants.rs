use std::time::Duration;

// limit used to avoid overwhelming wayland connection
pub const SENT_DAMAGE_LIMIT: usize = 256;

/// Maximum number of unacknowledged `RawBuffer` frames `wprsd` allows in flight
/// to `wprsc` at once. Allows encode + wire + decode pipelining at 60 FPS over
/// typical WAN RTTs while bounding queued wire memory under link stalls.
pub const MAX_IN_FLIGHT_FRAMES: usize = 3;

/// Interval between `Event::Heartbeat` probes sent by `wprsc` and stall checks
/// run by `wprsd`.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Duration an unacknowledged heartbeat may remain outstanding before `wprsc`
/// suppresses blind keyboard and pointer input until the link recovers.
pub const STALE_INPUT_THRESHOLD: Duration = Duration::from_secs(5);

/// Duration without peer response or activity before `wprsc` or `wprsd` treats
/// the connection as stalled and closes it cleanly.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(20);

/// Interval between periodic sweeps that compress dormant surface shadow
/// buffers on `wprsd`.
pub const DORMANT_SHADOW_SWEEP_INTERVAL: Duration = Duration::from_secs(5);

/// Idle duration without active buffer updates before a large surface shadow
/// buffer is eligible for dormant Zstd compression.
pub const DORMANT_SHADOW_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum number of dormant shadow buffers attempted per sweep tick so
/// aging-out multi-window sessions never stall the Wayland event loop.
pub const MAX_DORMANT_COMPRESS_PER_TICK: usize = 2;
