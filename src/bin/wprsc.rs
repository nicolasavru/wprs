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

use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use bpaf::Parser;
use optional_struct::optional_struct;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use smithay::reexports::calloop::EventLoop;
use smithay::reexports::calloop::channel::Event;
use smithay::reexports::calloop::timer::TimeoutAction;
use smithay::reexports::calloop::timer::Timer;
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::reexports::client::ConnectError;
use smithay_client_toolkit::reexports::client::Connection;
use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
use tracing::Level;
use wprs::args;
use wprs::args::CompressionKind;
use wprs::args::Config;
use wprs::args::JpegQuality;
use wprs::args::OptionalConfig;
use wprs::args::SerializableLevel;
use wprs::client::ClientOptions;
use wprs::client::HeartbeatTickAction;
use wprs::client::WprsClientState;
use wprs::constants;
use wprs::control_server;
use wprs::prelude::*;
use wprs::serialization::CompressionMode;
use wprs::serialization::JpegSubsamp;
use wprs::serialization::Serializer;
use wprs::utils;

#[optional_struct]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct WprscConfig {
    // Skip serializing fields which aren't ever useful to put into a config
    // file.
    #[serde(skip_serializing)]
    print_default_config_and_exit: bool,
    #[serde(skip_serializing)]
    config_file: PathBuf,
    pub socket: PathBuf,
    pub control_socket: PathBuf,
    // Optional fields don't get wrapped unless we specify it ourselves
    #[optional_wrap]
    pub log_file: Option<PathBuf>,
    pub stderr_log_level: SerializableLevel,
    pub file_log_level: SerializableLevel,
    pub log_priv_data: bool,
    pub title_prefix: String,
    pub compression_mode: CompressionKind,
    pub jpeg_quality: JpegQuality,
    pub jpeg_subsamp: JpegSubsamp,
}

impl WprscConfig {
    pub fn resolved_compression_mode(&self) -> CompressionMode {
        match self.compression_mode {
            CompressionKind::Lossless => CompressionMode::Lossless,
            CompressionKind::Hybrid => CompressionMode::Hybrid {
                quality: self.jpeg_quality.get(),
                subsamp: self.jpeg_subsamp,
            },
        }
    }
}

impl Default for WprscConfig {
    fn default() -> Self {
        Self {
            print_default_config_and_exit: false,
            config_file: args::default_config_file("wprsc"),
            socket: args::default_socket_path(),
            control_socket: args::default_control_socket_path("wprsc"),
            log_file: None,
            stderr_log_level: SerializableLevel(Level::INFO),
            file_log_level: SerializableLevel(Level::TRACE),
            log_priv_data: false,
            title_prefix: String::new(),
            compression_mode: CompressionKind::default(),
            jpeg_quality: JpegQuality::default(),
            jpeg_subsamp: JpegSubsamp::default(),
        }
    }
}

impl Config for WprscConfig {
    fn config_file(&self) -> PathBuf {
        self.config_file.clone()
    }
}

impl OptionalWprscConfig {
    fn args_parser() -> impl Parser<Self> {
        let print_default_config_and_exit = args::print_default_config_and_exit();
        let config_file = args::config_file();
        let socket = args::socket();
        let control_socket = args::control_socket();
        let log_file = args::log_file();
        let stderr_log_level = args::stderr_log_level();
        let file_log_level = args::file_log_level();
        let log_priv_data = args::log_priv_data();
        let title_prefix = args::title_prefix();
        let compression_mode = args::compression_mode();
        let jpeg_quality = args::jpeg_quality();
        let jpeg_subsamp = args::jpeg_subsamp();
        bpaf::construct!(Self {
            print_default_config_and_exit,
            config_file,
            socket,
            control_socket,
            log_file,
            stderr_log_level,
            file_log_level,
            log_priv_data,
            title_prefix,
            compression_mode,
            jpeg_quality,
            jpeg_subsamp,
        })
    }
}

impl OptionalConfig<WprscConfig> for OptionalWprscConfig {
    fn parse_args() -> Self {
        Self::args_parser().to_options().run()
    }

    fn print_default_config_and_exit(&self) -> Option<bool> {
        self.print_default_config_and_exit
    }

    fn config_file(&self) -> Option<PathBuf> {
        self.config_file.clone()
    }
}

fn main() -> Result<()> {
    let config = args::init_config::<WprscConfig, OptionalWprscConfig>();
    let compression_mode = config.resolved_compression_mode();
    args::set_log_priv_data(config.log_priv_data);
    utils::configure_tracing(
        config.stderr_log_level.0,
        config.log_file,
        config.file_log_level.0,
    )
    .location(loc!())?;
    utils::exit_on_thread_panic();

    let conn = Connection::connect_to_env().map_err(|e| match e {
        // give a more helpful/actionable message, since people who aren't familiar with wayland will run into this
        ConnectError::NoCompositor => {
            anyhow!("{e}, make sure you're running wprs from a wayland desktop environment")
        },
        _ => anyhow!(e),
    })?;

    let (globals, event_queue) = registry_queue_init(&conn)?;

    fs::create_dir_all(config.socket.parent().location(loc!())?).location(loc!())?;
    let mut serializer = Serializer::new_client(&config.socket).with_context(loc!(), || {
        format!(
            "Serializer unable to connect to socket {:?}.",
            config.socket
        )
    })?;
    let reader = serializer.reader().location(loc!())?;

    let options = ClientOptions {
        title_prefix: config.title_prefix,
        compression_mode,
    };
    let mut state = WprsClientState::new(
        event_queue.handle(),
        globals,
        conn.clone(),
        serializer,
        options,
    )
    .location(loc!())?;

    let mut event_loop = EventLoop::try_new()?;
    let reader_loop_signal = event_loop.get_signal();

    event_loop
        .handle()
        .insert_source(
            reader,
            move |event, _metadata, state: &mut WprsClientState| match event {
                Event::Msg((_, msg)) => state.handle_request(msg),
                Event::Closed => {
                    reader_loop_signal.stop();
                },
            },
        )
        .unwrap();

    let heartbeat_loop_signal = event_loop.get_signal();
    event_loop
        .handle()
        .insert_source(
            Timer::from_duration(constants::HEARTBEAT_INTERVAL),
            move |_deadline, _metadata, state: &mut WprsClientState| match state
                .handle_heartbeat_tick(Instant::now())
            {
                HeartbeatTickAction::Continue => {
                    TimeoutAction::ToDuration(constants::HEARTBEAT_INTERVAL)
                },
                HeartbeatTickAction::TimedOut => {
                    heartbeat_loop_signal.stop();
                    TimeoutAction::Drop
                },
            },
        )
        .unwrap();

    {
        let capabilities = state.capabilities.clone();
        control_server::start(config.control_socket, move |input: &str| {
            Ok(match input {
                // TODO: make the input use json when we have more commands
                "caps" => serde_json::to_string(&capabilities.get(Duration::from_secs(5)))
                    .expect("a map with non-string keys was added to Capabilities"),
                _ => {
                    bail!("Unknown command: {input:?}")
                },
            })
        })
        .location(loc!())?;
    }

    WaylandSource::new(conn, event_queue)
        .insert(event_loop.handle())
        .location(loc!())?;

    event_loop.run(None, &mut state, |_| {}).location(loc!())?;

    if state.timed_out() {
        bail!("connection to wprsd timed out");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use optional_struct::Applicable;
    use ron::Options;
    use ron::extensions::Extensions;

    use super::*;

    #[test]
    fn test_wprsc_config_compression_cli_and_ron() {
        let default_cfg = WprscConfig::default();
        assert_eq!(
            default_cfg.resolved_compression_mode(),
            CompressionMode::Lossless
        );

        // `--print-default-config-and-exit` output must roundtrip through `OptionalWprscConfig`.
        let printed =
            ron::ser::to_string_pretty(&default_cfg, ron::ser::PrettyConfig::default()).unwrap();
        let parsed_opt: OptionalWprscConfig = Options::default()
            .with_default_extension(Extensions::IMPLICIT_SOME)
            .from_str(&printed)
            .unwrap();
        let mut applied = WprscConfig::default();
        parsed_opt.apply_to(&mut applied);
        assert_eq!(applied, default_cfg);

        // Hybrid RON overrides apply cleanly; invalid quality fails loudly.
        let hybrid_ron = "(compression_mode: Hybrid, jpeg_quality: 85, jpeg_subsamp: Sub420)";
        let hybrid_opt: OptionalWprscConfig = Options::default()
            .with_default_extension(Extensions::IMPLICIT_SOME)
            .from_str(hybrid_ron)
            .unwrap();
        let mut hybrid_cfg = WprscConfig::default();
        hybrid_opt.apply_to(&mut hybrid_cfg);
        assert_eq!(
            hybrid_cfg.resolved_compression_mode(),
            CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            }
        );

        for bad_ron in ["(jpeg_quality: 0)", "(jpeg_quality: 101)"] {
            assert!(
                Options::default()
                    .with_default_extension(Extensions::IMPLICIT_SOME)
                    .from_str::<OptionalWprscConfig>(bad_ron)
                    .is_err()
            );
        }

        // CLI flag parsing via `OptionalWprscConfig::args_parser`.
        let cli_opt = OptionalWprscConfig::args_parser()
            .to_options()
            .run_inner(
                &[
                    "--compression-mode",
                    "hybrid",
                    "--jpeg-quality",
                    "88",
                    "--jpeg-subsamp",
                    "422",
                ][..],
            )
            .unwrap();
        let mut cli_cfg = WprscConfig::default();
        cli_opt.apply_to(&mut cli_cfg);
        assert_eq!(
            cli_cfg.resolved_compression_mode(),
            CompressionMode::Hybrid {
                quality: 88,
                subsamp: JpegSubsamp::Sub422,
            }
        );
        assert!(
            OptionalWprscConfig::args_parser()
                .to_options()
                .run_inner(&["--jpeg-quality", "0"][..])
                .is_err()
        );
        assert!(
            OptionalWprscConfig::args_parser()
                .to_options()
                .run_inner(&["--jpeg-quality", "101"][..])
                .is_err()
        );
    }
}
